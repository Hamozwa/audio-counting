import os
import csv

output_dir = "/scratch/local/ssd/hani/bbc_clocks/"
csv_path = "preprocessing/BBCSoundEffects.csv"

#get zip files
zip_dir = output_dir + "zips/"
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "clock" in row["CDName"].lower():
            id = row["location"].removesuffix(".wav")
            save_site = f"https://sound-effects-media.bbcrewind.co.uk/zip/{id}.wav.zip?download&rename=BBC_Clocks--Co_{id}"

            wget_command = f"wget -O {os.path.join(zip_dir, id)}.zip '{save_site}'"
            os.system(wget_command)

#unzip files
sound_dir = output_dir + "audio/"
for file in os.listdir(zip_dir):
    if file.endswith(".zip"):
        unzip_command = f"unzip -o {os.path.join(zip_dir, file)} -d {sound_dir}"
        os.system(unzip_command)