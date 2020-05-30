
# Writing up work of last week for Daniel 30 May 2020
Main work on VAE-VQ-2
see 
### main_vqvae-2.py
1. create 10s segments of curated normal segments, 300 segments which are listed in data/dataset_csv/sample_file.csv
loads them up into model and trains on them

2 . computes loss and metrics of spectrum and histogram loss


- what does it mean to reconstruct? 
original -> latent -> reconstruct from encoding then compares


### Next steps:
- Part A:
  - re-run prior model nn-vae on currated for comparison show
  - check if fourioer metric is computing correctly by compting more often or if some artifact of computaiton is a problem

- Part B: generate samples of normal EEG. 
  - generate condiational samples of EEG based upon adjance channels 




- Backburner: 
  - future conditional generation based upon age.



