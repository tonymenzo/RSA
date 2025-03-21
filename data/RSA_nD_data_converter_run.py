import subprocess

# Define the parameters
aLund = 0.72  # Example value for aLund
bLund = 0.88  # Example value for bLund
sigma = 0.335  # Example value for sigma
nEvent = 10000  # Example value for nEvent

# Use formatted string literals (f-strings) to automatically generate the paths
# filename_base = f"/global/homes/l/ljpuslar/RSA/releases/examples/pgun_qqbar_finalTwo_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
filename_base = f"/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
print(filename_base) 

# Generate the paths based on the filename base
hadron_PATH = f"{filename_base}_hadrons.txt"
acceptReject_PATH = f"{filename_base}_accept_reject_z.txt"
mT2_PATH = f"{filename_base}_mT2.txt"
fPrel_PATH = f"{filename_base}_fPrel.txt"
id_PATH = f"{filename_base}_pid.txt"

# Print the paths to check
print("Hadron Path: ", hadron_PATH)
print("AcceptReject Path: ", acceptReject_PATH)
print("mT2 Path: ", mT2_PATH)
print("fPrel Path: ", fPrel_PATH)
print("id Path: ", id_PATH)

# Use formatted string literals (f-strings) to automatically generate the paths
filename_base = f"/global/homes/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"

# Generate the paths based on the filename base
write_hadron_PATH = f"{filename_base}_hadrons.npy"
write_fPrel_PATH = f"{filename_base}_fPrel.npy"
write_id_mT2_acceptReject_PATH = f"{filename_base}_id_mT2_accept_reject_z.npy" #-> (id_old, id_new, mT2, accept_reject_chain)

# Define the arguments to be passed
args = [
    "python", "RSA_nD_data_converter.py",  # Call the Python script you want to execute
    "--data_path_accept_reject", acceptReject_PATH,
    "--data_path_fPrel", fPrel_PATH,
    "--data_path_mT2", mT2_PATH,
    "--data_path_hadrons", hadron_PATH,
    "--data_path_id", id_PATH,
    "--write_path_id_mT2_accept_reject", write_id_mT2_acceptReject_PATH,
    "--write_path_fPrel", write_fPrel_PATH,
    "--write_path_hadrons", write_hadron_PATH,
    "--print_details", "True"
]

# Call the script with the arguments
subprocess.run(args)