# RNA-Compound Interaction Classification

## Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Lyttr/RNA_Compound-interaction-classification.git
   cd RNA_Compound-interaction-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate dataset:
   ```bash
   python src/data/dataset.py
   ```

4. Generate feature embeddings:
   ```bash
   python src/scripts/embedding_generate.py
   ```

5. Run baselines:
   ```bash
   python src/baselines/baseline_tokenizer.py
   python src/baselines/baseline_transformer.py
   python src/baselines/baseline_meanpooling.py
   ```

6. Train the model:
   ```bash
   python src/scripts/train.py --train_path datasets/trainset.pt --test_path datasets/testset.pt --output_dir results/test --project test_project --run_name test_run
   ```
