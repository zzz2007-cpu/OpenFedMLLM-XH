pip install -U huggingface_hub
hf download derek-thomas/ScienceQA --repo-type dataset --local-dir ScienceQA
python scripts/filter_scienceqa_image_present.py
python scripts/data/build_federated_scienceqa_chi.py --input_dir ScienceQA/image_present/data --output_dir data/scienceqa/federated_chi --num_clients 10 --settings L0_M0 L1_M0 L0_M1 L1_M1 L2_M2 --split_target subject --alpha_l1 0.5 --alpha_l2 0.1 --seed 42