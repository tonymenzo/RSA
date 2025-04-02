import subprocess

# # Use formatted string literals (f-strings) to automatically generate the paths
# # filename_base = f"/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
# filename_base = f"/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_standarda_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
# filename_base = f"/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_monashaD0_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.88_bS1_bC0.88_bB0.88_bH0.88_sigma_0.335_N_1.0e+03_pid"

# Define the parameters
aLund = 0.68 # Example value for aLund
bLund = 0.98  # Example value for bLund
sigma = 0.335  # Example value for sigma
nEvent = 100000  # Example value for nEvent

# Define parameters
aExtraDQuark = 0.15
aExtraUQuark = 0
aExtraSQuark = 0
aExtraCquark = 0
aExtraBquark = 0
aExtraDiquark = 0.97


bNonstandardD = 0.98 - 0.3
bNonstandardU = 0.98
bNonstandardS = 0.98
bNonstandardC = 0.98
bNonstandardB = 0.98
bNonstandardH = 0.98


def format_double(value):
    # if value == 0:
    #     return f"{value}"
    if value == int(value):
        return f"{int(value)}"
    else:
        return f"{value}"


# Construct the filename string with enforced double precision
filename_base = (
    # f"/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/"
    f"/pscratch/sd/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/"
    f"pgun_uubar__"
    f"a{format_double(aLund)}_"    
    f"b{format_double(bLund)}_"    
    f"aD{format_double(aExtraDQuark)}_"
    f"aU{format_double(aExtraUQuark)}_"
    f"aS{format_double(aExtraSQuark)}_"
    f"aC{format_double(aExtraCquark)}_"
    f"aB{format_double(aExtraBquark)}_"
    f"aH{format_double(aExtraDiquark)}_"
    f"bD{format_double(bNonstandardD)}_"
    f"bU{format_double(bNonstandardU)}_"
    f"bS{format_double(bNonstandardS)}_"
    f"bC{format_double(bNonstandardC)}_"
    f"bB{format_double(bNonstandardB)}_"
    f"bH{format_double(bNonstandardH)}_"
    f"sigma_{format_double(sigma)}_"  
    f"N_{nEvent:.1e}"  # Keep nEvent in scientific notation
)

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
# filename_base = f"/global/homes/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
# filename_base = f"/global/homes/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_standard_a_{aLund}_b_{bLund}_sigma_{sigma}_N_{nEvent:.1e}"
filename_base = (
    f"/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/"
    # f"/global/homes/l/ljpuslar/RSA/RSA/data/structured_data/"
    f"pgun_uubar__"
    f"a{aLund}_"
    f"b{bLund}_"
    f"aD{aExtraDQuark}_"
    f"aU{aExtraUQuark}_"
    f"aS{aExtraSQuark}_"
    f"aC{aExtraCquark}_"
    f"aB{aExtraBquark}_"
    f"aH{aExtraDiquark}_"
    f"bD{bNonstandardD}_"
    f"bU{bNonstandardU}_"
    f"bS{bNonstandardS}_"
    f"bC{bNonstandardC}_"
    f"bB{bNonstandardB}_"
    f"bH{bNonstandardH}_"
    f"sigma_{sigma}_"
    f"N_{nEvent:.1e}"  # Formats nEvent in scientific notation
)

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


# '/global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_monash_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.88_bS1.0_bC0.88_bB0.88_bH0.88_sigma_0.335_N_1.0e+03_accept_reject_z.txt'
# /global/homes/l/ljpuslar/RSA/pythia8312/examples/pgun_data/model/pgun_uubar_monash_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.88_bS1_bC0.88_bB0.88_bH0.88_sigma_0.335_N_1.0e+03_accept_reject_z.txt