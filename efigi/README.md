### Readme with help for each file

`cgan.py`: conditional GAN training file. Parameters:
- batch size
- redshift (bool)
- data directory
- epochs
- loss (gan type)

`python cgan.py --BATCH_SIZE=64 --REDSHIFT=1 --DATA_DIR=/mnt/data1/images/efigi/ --EPOCHS=1000 --LOSS=wgan`

___

`dataset_stats.py`: prints out some stats about the EFIGI dataset
- data directory

`python dataset_stats.py --DATA_DIR=/mnt/data1/images/efigi/`

___


`generate_test_galaxies.py`: Takes the entire test set attributes and generates from those with random z vectors.
Each image saved out is the original test image on the left, then the next 5 columns are the generated ones.

`python generate_test_galaxies.py --CHECKPOINT_DIR=checkpoints/LOSS_wgan/REDSHIFT_True/ --OUTPUT_DIR=output/galaxies_out/ --REDSHIFT=1 --DATA_DIR=/mnt/data1/images/`

___

`generate_specific.py`: Generates `n` images all given different z vectors but with a user specified attribute. Useful
for testing out different redshifts.

___


