import fm
from pathlib import Path
data_dir = '../input/rnafm-tutorial/'
fm_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
print(fm_model.args)
model=fm.BioBertModel(fm_model.args,alphabet)