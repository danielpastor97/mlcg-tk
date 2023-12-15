# Input Training Data Generation Pipeline
-----------------------------------------

### How To

To prepare simulation data for eventual use in training a transferable CG forcefield, these data must first be mapped to the CG resolution and then processed to incorporate prior energy terms. This is done here in  two steps: first, in `gen_input_data.py` the atomistic simualtions are loaded and mapped to the lower resolution, followed by the construction of the neighbourlists specific to the chosen prior energy model in the same step. Next, the outputs are fed to `produce_delta_forces.py` along with the prior model so that delta forces can be calculated. The result is a set of CG coordinates, embeddings, and delta forces which can be used to train a neural network as well as the neighbourlists associated with each molecule which can be fitted to produce new prior models, if needed. In the details below, input files are provided as an example.

#### 1) Loading and processing all-atom simulation data

Command:

`python gen_input_data.py --data_dict cath_data_dict.yaml --names cath_names.yaml --prior_dict prior_dict.yaml`

This procedure will loop over all of the datasets provided in the `--data_dict` and the corresponding sample names in `--names`. For each instance, it will load the atomistic coordinates, forces, and structures and map these to a lower resolution specified in the input file (this allows for various resolutions and CG embeddings to be used). Then, using the keys of the `--prior_dict`, the script will generate a neighbourlist for each molecule, so long as the prior terms and their corresponding functions specified in the input file are defined in `prior_terms.py`. 

#### 2) Producing delta forces

Command:

`python produce_delta_forces.py --data_dict cath_data_dict.yaml --names cath_names.yaml --prior_model /import/a12/users/aguljas/mlcg_base_cg_data/test_pipeline/priors/full_prior_model_cgschnet.pt`

This procedure will load the prior model specified by `--prior_model` and then once again loop over all datasets and sample names provided. It will then calculate and remove the baseline forces using the coordinates, forces, embeddings, and neighbourlists created in the previous step. It will then save the delta forces which can then be used for training. At this point, the delta force calculation can only be run locally.


