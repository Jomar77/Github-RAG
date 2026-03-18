import io
import zipfile
import requests
import frontmatter



def main():
    url = 'https://codeload.github.com/DataTalksClub/faq/zip/refs/heads/main'
    resp = requests.get(url)


if __name__ == "__main__":
    main()
