gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command "git clone https://github.com/joytianya/t5_tf.git "
gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command "
cd t5_tf/t5x_scripts/t5x/
python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
" 

gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command "pip3 install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command "
pip3 install tensorflow-text==2.11.0
"
gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command 'python3 -c "import jax; print(jax.devices()) "'
