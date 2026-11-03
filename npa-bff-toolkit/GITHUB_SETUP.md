# Publish this folder to GitHub

1. Open https://github.com/new and create a repository named `npa-bff-toolkit` under your account. Choose **Private** initially if you want to review the code before publication. **Do not** initialize the new repository with a README, .gitignore or license, as these are managed locally.
2. Open a terminal in the extracted `npa-bff-toolkit` folder.
3. Run the following commands (GitHub CLI / browser authentication may be needed for the push):

```bash
git init
git add .
git commit -m "Initial NPA and BFF research toolkit"
git branch -M main
git remote add origin https://github.com/pritam-roy-99/npa-bff-toolkit.git
git push -u origin main
```

4. On GitHub, go to **Actions** to check that the automated tests run.
5. Before switching the repository to Public, review the code, add a license if you want to grant reuse permissions, and make sure no private data or tokens were committed.

**Alternative without Git:** Create the empty repository, choose **uploading an existing file**, and drag the unpacked contents into the web interface. Directories beginning with a dot (such as `.github/`) may be harder to include through a browser, so a Git push is recommended.
