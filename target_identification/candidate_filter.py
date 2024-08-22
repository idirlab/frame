# Add Hydra dir to python path
import sys
sys.path.append('YOUR/PATH/TO/THIS/REPO')

from safetensors.torch import load_model
import pandas as pd
import Hydra.config as config
from transformers import AutoTokenizer
from Hydra.models.gold_target_frame_id_token import RobertaForFrameIdentificationWithTargetToken
from datetime import datetime
from Hydra.utils.train import compute_metrics
import os
from transformers import Trainer, TrainingArguments
from Hydra.utils.target_identification.data import TargetIdentificationDataset
from scipy.special import softmax
from Hydra.utils.target_identification.data import load_dataset

os.environ['WANDB_API_KEY'] = ''
os.environ['WANDB_PROJECT'] = ''
os.environ['WANDB_LOG_MODEL'] = ''

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Load dataset
print(f'Loading data...')
train_data = load_dataset('train_fulltext_data_tokenized.json', tokenizer)
dev_data = load_dataset('dev_fulltext_data_tokenized.json', tokenizer)
test_data = load_dataset('test_fulltext_data_tokenized.json', tokenizer)

# Create datasets
print(f'Processing datasets...')
train_dataset = TargetIdentificationDataset(train_data, tokenizer)
dev_dataset = TargetIdentificationDataset(dev_data, tokenizer)
test_dataset = TargetIdentificationDataset(test_data, tokenizer)

# Load model
model = RobertaForFrameIdentificationWithTargetToken.from_pretrained('roberta-base', num_labels=2)
# load_model(model, './models/candidate_filter/model.safetensors')

# Unique run name
run_name = f'target_filter-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Save run config
# with open(f'{config.model_dir}/frame_identification/{run_name}_args.txt', 'w') as f:
#     f.write(str(args))

# Set up training arguments
training_args = TrainingArguments(
    report_to = 'wandb',
    run_name=run_name,
    output_dir= f'{config.model_dir}/target_identification/{run_name}',
    num_train_epochs=3,
    per_device_train_batch_size=36, # Fits on v100 and T4
    per_device_eval_batch_size=36,
    warmup_ratio=0.05,
    weight_decay=0.001,
    logging_dir=config.logs_dir,
    logging_steps=20,
    evaluation_strategy='steps',
    eval_steps=0.1,
    save_steps=0.1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_strategy='steps',
    lr_scheduler_type='cosine',
    learning_rate=5e-5,
    seed=0,
    data_seed=0
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=compute_metrics,     # metrics to compute
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset             # evaluation dataset
)

# Train model
train_result = trainer.train()

# Log training metrics to file
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

print('Evaluating model on dev set...')
# Evaluate model
trainer.evaluate()

print('Evaluating model on test set...')
# Predict on test set
predictions = trainer.predict(test_dataset)
print(predictions.metrics)

# Softmax predictions
preds = softmax(predictions.predictions, axis=-1)

# Save prediction scores with inputs and labels
predictions_df = pd.DataFrame({'sentence': test_dataset.dataset.sentence,
                            'target_token_span': test_dataset.target_token_span.tolist(),
                            'label': test_dataset.labels,
                            'preds': preds.tolist()})

predictions_df.to_json(f'{config.model_dir}/target_identification/{run_name}_predictions.json')
