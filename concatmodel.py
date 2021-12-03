from torch import nn

from transformers import BertForSequenceClassification
from mlp import MLP
import torch
from torch.nn import CrossEntropyLoss, MSELoss


class BertConcatFeatures(BertForSequenceClassification):
    """
    A model for classification or regression which combines text, categorical,
    and numerical features. The text features are processed with BERT. All
    features are concatenated into a single vector, which is fed into an MLP
    for final classification / regression.

    This class expects a transformers.BertConfig object, and the config object
    needs to have three additional properties manually added to it:
      `text_feat_dim` - The length of the BERT vector.
      `cat_feat_dim` - The number of categorical features.
      `numerical_feat_dim` - The number of numerical features.
    """

    def __init__(self, config):

        # ====================
        #     BERT Setup
        # ====================

        # Call the constructor for the huggingface `BertForSequenceClassification`
        # class, which will do all of the BERT-related setup. The resulting BERT
        # model is stored in `self.bert`.
        super().__init__(config)

        # ==================================
        #     Feature Combination Setup
        # ==================================

        # Store the number of labels, which tells us whether this is a
        # classification or regression task.
        self.num_labels = config.num_labels

        # Calculate the combined vector length.
        combined_feat_dim = config.text_feat_dim + \
                            config.cat_feat_dim + \
                            config.numerical_feat_dim

        # Create a batch normalizer for the numerical features.
        self.num_bn = nn.BatchNorm1d(config.numerical_feat_dim)

        # ====================
        #     MLP Setup
        # ====================

        # To setup the MLP, we need to specify the number of layers and the
        # number of neurons in each layer. The MultiModal-Toolkit has a formula
        # for picking these dimensions. Each layer of the MLP has 1/4th the
        # number of neurons as the previous one.

        # Dimensions of each MLP layer.
        dims = []

        # Starting with the combined feature vector length...
        dim = combined_feat_dim

        # Keep dividing by 4 until we drop below the number of outputs the MLP
        # needs to have.
        while True:

            # Divide by 4 and truncate to an integer.
            dim = dim // 4

            # If the resulting layer size would be smaller than the number of
            # outputs, then we're done.
            if dim <= self.num_labels:
                break

            # Otherwise, store this as the next layer size.
            dims.append(int(dim))

        # Print out the resulting MLP.
        print('MLP layer sizes:')
        print('  Input:', combined_feat_dim)
        print('  Hidden:', dims)
        print('  Output:', self.num_labels)
        print('')

        # Construct the MLP, specifying the number of inputs, outputs, and the
        # layer sizes.
        self.mlp = MLP(combined_feat_dim,
                       self.num_labels,
                       num_hidden_lyr=len(dims),
                       dropout_prob=0.1,
                       hidden_channels=dims,
                       bn=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            class_weights=None,
            output_attentions=None,
            output_hidden_states=None,
            numerical_feats=None
    ):
        r"""
        Perform a forward pass of our model.

        This has the same inputs as `forward` in `BertForSequenceClassification`,
        but with two extra parameters:
          `cat_feats` - Tensor of categorical features.
          `numerical_feats` - Tensor of numerical features.
        """

        # ====================
        #        BERT
        # ====================

        # Run the text through the BERT model. Invoking `self.bert` returns
        # outputs from the encoding layers, and not from the final classifier.
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # outputs[0] - All of the output embeddings from BERT
        # outputs[1] - The [CLS] token embedding, with some additional "pooling"
        #              done.
        cls = outputs[1]

        # Apply dropout to the CLS embedding.
        cls = self.dropout(cls)

        # ==========================
        #    Concatenate Features
        # ==========================

        # Apply batch normalization to the numerical features.
        numerical_feats = self.num_bn(numerical_feats)

        # Object sizes:
        #             cls   [batch size  x   768]
        # numerical_feats   [batch size  x   # numerical features]
        #       cat_feats   [batch size  x   # categorical features]

        # Simply concatenate everything into one vector.
        # For example, if we have 4 categ. and 3 numer. features, then the
        # result has 768 + 4 + 3 = 775 features.
        combined_feats = torch.cat((cls, numerical_feats),
                                   dim=1)

        # ====================================
        #    Output Classifier / Regression
        # ====================================

        # Run the the samples through the MLP.
        logits = self.mlp(combined_feats)

        # TODO - Not sure what's going on with the outputs...

        if type(logits) is tuple:
            logits, classifier_layer_outputs = logits[0], logits[1]

        else:  # simple classifier
            classifier_layer_outputs = [combined_feats, logits]

        # =================
        #       Loss
        # =================

        # Calculate loss, but only if labels were provided.
        # (Labels aren't provided at test time).
        if labels is not None:

            # Regression
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()

                labels = labels.float()

                loss = loss_fct(logits.view(-1), labels.view(-1))

            # Classification
            else:
                loss_fct = CrossEntropyLoss(weight=class_weights)

                labels = labels.long()

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # If no labels are provided, set the loss to 'None'.
        else:
            loss = None

        # Put the results into a Dictionary to return.
        results = {'loss': loss,
                   'logits': logits,
                   'classifier_layer_outputs': classifier_layer_outputs}

        return results