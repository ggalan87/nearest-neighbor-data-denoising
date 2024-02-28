from bs4 import BeautifulSoup
import requests

file = '/home/ggalan/tmp/umap_embedding_single.html'

with open(file) as f:
    soup = BeautifulSoup(f, 'html.parser')

links = soup.find_all('a')
print(links)
