# RNA-Compound interaction classification
# Steps
1. Clone the repository:
```bash
git clone https://github.com/Lyttr/RNA_Compound-interaction-classification.git
cd RNA_Compound-interaction-classification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Generate dataset
```bash
python dataset.py
```
4. Generate features embeddings
```bash
python embedding_generate.py
```
embeddings of baselines
```bash
python baseline_tokenizer.py
python baseline_transformer.py
python baseline_meanpooling.py
```
5. Train the model
```bash
python train.py --train_path datasets/trainset.pt --test_path datasets/testset.pt --output_dit results/test --project test_project --run_name test_run
```
