import torch
from transformers import BertConfig, BertTokenizer
from concatmodel import BertConcatFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import random
import numpy as np
import datetime
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch.nn.functional as F

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

col_info = {}
col_info['text_cols'] = ["cleaned_comment"]
col_info["num_cols"] = ["cogency", "effectiveness", "reasonableness", "overall"]
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
train_df = pd.read_csv("5fold/split0/train.csv", sep="\t")
val_df = pd.read_csv("5fold/split0/val.csv", sep="\t")
attention_masks = []
max_len = 512
print('Encoding {:,} text samples...'.format(len(train_df)))

num_done = 0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# For each of the samples...
for (row_i, row) in train_df.iterrows():

    # Update every 2k samples.
    if ((num_done % 2000) == 0):
        print('  {:>6,}'.format(num_done))

    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        row['cleaned_comment'],  # Sentence to encode.
        max_length=max_len,  # Pad & truncate all sentences.
        truncation=True,
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    num_done += 1
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

print('DONE.')

labels = torch.tensor(train_df['label'].values)
# Tensor for numerical features.
numerical_feats = torch.tensor(train_df[col_info['num_cols']].values.astype('float'), dtype=torch.float32)
train_dataset = TensorDataset(input_ids, numerical_feats, attention_masks, labels)

input_ids = []
attention_masks = []

for (row_i, row) in val_df.iterrows():

    # Update every 2k samples.
    if ((num_done % 2000) == 0):
        print('  {:>6,}'.format(num_done))

    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        row['cleaned_comment'],  # Sentence to encode.
        max_length=max_len,  # Pad & truncate all sentences.
        truncation=True,
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    num_done += 1
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

print('DONE.')

labels = torch.tensor(val_df['label'].values)
# Tensor for numerical features.
numerical_feats = torch.tensor(val_df[col_info['num_cols']].values.astype('float'), dtype=torch.float32)
val_dataset = TensorDataset(input_ids, numerical_feats, attention_masks, labels)
if torch.cuda.is_available():

    # Print out what GPU we've got.
    print('There are %d GPU(s) available:\n' % torch.cuda.device_count())
    print('    ', torch.cuda.get_device_name(0))

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

# If not, you could use the CPU, but this isn't recommended!
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

from transformers import BertConfig

# We'll need to a use a "BertConfig" object from the transformers library to
# specify our parameters.

# First, specify the ordinary BERT parameters by taking them from the
# 'bert-base-uncased' model.
# Also set the number of labels.
config = BertConfig.from_pretrained(
    'bert-base-uncased',
    num_labels=5,  # The number of output labels--1 for regression.
)

# To set up the MLP, we need to know the combined vector length that will be
# sent into it.

# Pass in the number of numerical and categorical features.
config.numerical_feat_dim = numerical_feats.size()[1]
config.cat_feat_dim = 0

# Pass in the size of the text embedding.
# The text feature dimension is the "hidden_size" parameter which
# comes from BertConfig. The length is 768 in BERT-base (and most other BERT
# models).
config.text_feat_dim = config.hidden_size  # 768

# Now we're ready to do the actual set up of our model! Note that we're passing
# in the config object here.
model = BertConcatFeatures.from_pretrained(
    "bert-base-uncased",
    config=config
)

# Tell pytorch to run this model on the GPU.
desc = model.cuda()

# Larger batch sizes tend to be better, and we can fit this in memory.
batch_size = 16

# This is the learning rate specified in Ken's configuration
learning_rate = 5e-3

# Number of training epochs.
epochs = 20

# Print out the max_len for reference. To change this, you'd need to set it
# back in section 3 prior to the tokenization and encoding of the text.
print('Using maximum sequence length:', max_len)

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)
# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=1e-8
                  )
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        b_input_ids = batch[0].to(device)
        b_numer_feats = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # Clear prior gradients.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # In PyTorch, calling `model` will in turn call the model's `forward`
        # function and pass down the arguments.
        # This will return the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.
        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       numerical_feats=b_numer_feats)

        loss = result['loss']
        logits = result['logits']

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    # Include the batch size so that we are looking at the average per-sample
    # loss.
    avg_train_loss = total_train_loss / (len(train_dataloader) * batch_size)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_loss = 0
    total_eval_f1 = 0.0
    nb_eval_steps = 0
    predictions, true_labels = [], []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_numer_feats = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           numerical_feats=b_numer_feats)

        # Get the loss and "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like the
        # softmax.
        loss = result['loss']
        logits = result['logits']


        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        #logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        probs = F.softmax(logits, dim=-1).detach()
        # if you have more than one label
        y_pred = np.argmax(probs.cpu().numpy(), axis=1).flatten()

        # Store predictions and true labels
        predictions.append(y_pred)
        true_labels.append(label_ids)
        f1score = f1_score(y_true=label_ids, y_pred=y_pred, average="macro")
        total_eval_f1 += f1score

        # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / (len(validation_dataloader) * batch_size)
    avg_f1 = total_eval_f1 / (len(validation_dataloader) * batch_size)
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation F1: {0:.2f}".format(avg_f1))

    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. F1': avg_f1,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap (doesn't seem to work in Colab).
# df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
print(df_stats)
