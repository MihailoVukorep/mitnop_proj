# https://www.nist.gov/itl/products-and-services/emnist-dataset

mkdir datasets
cd datasets
#wget --show-progress https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip -P datasets/.
curl "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip" --output gzip.zip
unzip gzip.zip

