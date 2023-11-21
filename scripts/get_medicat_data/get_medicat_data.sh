echo "cd to /gscratch/scrubbed/briggs3/data"
cd /gscratch/scrubbed/briggs3/data

mkdir -p medicat/
cd medicat

echo "wget medicat data images..."
wget https://ai2-s2-medicat.s3.us-west-2.amazonaws.com/2020-10-05/medicat_release.tar.gz 

echo "decompressing medicat data..."
# uncompress tar file
tar -xzvf medicat_release.tar.gz 

echo "removing medicat tar file..."
# remove tar file
rm medicat_release.tar.gz


echo "finished!"