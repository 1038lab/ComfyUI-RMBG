"""Refresh bundled non-weight resources, recording immutable sources and hashes.

Run manually during development; never called by a node. Existing manifest pins
are reused. No weight files are downloaded.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import ssl
import urllib.parse
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1] / "models" / "resources"
REPOS = [
    "1038lab/RMBG-2.0", "1038lab/BiRefNet", "1038lab/BEN", "1038lab/BEN2",
    "1038lab/segformer_clothes", "1038lab/segformer_face",
    "1038lab/segformer_fashion", "1038lab/FeyNobg", "1038lab/SDMatte",
    "1038lab/GroundingDINO", "briaai/VRMBG-3.0",
    "google-bert/bert-base-uncased",
    "microsoft/Florence-2-base", "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large", "microsoft/Florence-2-large-ft",
    "thwri/CogFlorence-2.1-Large", "thwri/CogFlorence-2.2-Large",
]


class ResourceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if redirected is not None and urllib.parse.urlparse(new_url).hostname != "huggingface.co":
            redirected.remove_header("Authorization")
        return redirected


def read(url):
    context = ssl.create_default_context()
    # Keep certificate verification, allowing older enterprise proxy CA chains.
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), ResourceRedirect())
    with opener.open(request, timeout=60) as response:
        return response.read()


def vendor(repo):
    target = ROOT / repo
    manifest_path = target / "sources.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = previous.get("revision", "main")
    metadata = json.loads(read(f"https://huggingface.co/api/models/{repo}/revision/{revision}"))
    revision = metadata["sha"]
    records = []
    missing = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        path = Path(name)
        # Florence uses the architecture/processor already in models/florence2.
        # Only its variant config and tokenizer data are needed from upstream.
        if repo.startswith(("microsoft/Florence-", "thwri/CogFlorence-")) and name not in {
                "config.json", "tokenizer.json", "README.md"} and not path.name.upper().startswith("LICENSE"):
            continue
        if len(path.parts) > 1 and not (repo == "1038lab/SDMatte" and path.parts[0] in
                {"scheduler", "text_encoder", "tokenizer", "unet", "vae"}):
            continue
        if path.suffix not in {".json", ".py", ".txt", ".cfg", ".md"} and not path.name.upper().startswith("LICENSE"):
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
        try:
            data = read(url)
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                raise
            missing.append(name)
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append({"file": name, "url": url, "sha256": hashlib.sha256(data).hexdigest()})
    target.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"repo": repo, "revision": revision,
        "license": metadata.get("cardData", {}).get("license"), "files": records,
        "requires_access": missing}, indent=2) + "\n")
    suffix = f"; access required for {missing}" if missing else ""
    return f"{repo}: {len(records)} resources at {revision}{suffix}"


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for repo, result in zip(REPOS, pool.map(vendor, REPOS)):
            print(result, flush=True)
