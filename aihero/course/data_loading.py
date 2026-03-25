from __future__ import annotations

import io
import zipfile

import frontmatter
import requests


def read_repo_data(repo_owner: str, repo_name: str) -> list[dict[str, object]]:
    """Download and parse markdown files from a GitHub repository main branch."""
    prefix = "https://codeload.github.com"
    url = f"{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main"
    resp = requests.get(url)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download repository: {resp.status_code}")

    repository_data: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        for file_info in archive.infolist():
            filename = file_info.filename
            filename_lower = filename.lower()

            if not (filename_lower.endswith(".md") or filename_lower.endswith(".mdx")):
                continue

            try:
                with archive.open(file_info) as file_stream:
                    content = file_stream.read().decode("utf-8", errors="ignore")
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data["filename"] = filename
                    repository_data.append(data)
            except Exception as error:
                print(f"Error processing {filename}: {error}")
                continue

    return repository_data
