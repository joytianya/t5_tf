gcloud alpha compute tpus tpu-vm ssh clueaiv3-256-tf291-vm --zone europe-west4-a --worker all --command "
cd t5_tf/t5x_scripts/
git pull
sh run.sh
"
