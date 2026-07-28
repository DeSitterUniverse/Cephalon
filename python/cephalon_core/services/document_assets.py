from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shutil
import uuid

from .pdf_parser import PdfAsset


ASSET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class AssetTransaction:
    root: str
    doc_id: str
    staging: str
    active: str
    backup: str
    rows: list[tuple]
    promoted: bool = False

    @classmethod
    def prepare(cls, data_dir: str, doc_id: str, assets: list[PdfAsset]) -> "AssetTransaction":
        root = os.path.abspath(os.path.join(data_dir, "document-assets"))
        staging_root = os.path.join(root, ".staging")
        backup_root = os.path.join(root, ".backups")
        os.makedirs(staging_root, exist_ok=True)
        os.makedirs(backup_root, exist_ok=True)
        token = uuid.uuid4().hex
        staging = os.path.join(staging_root, f"{doc_id}-{token}")
        active = os.path.join(root, doc_id)
        backup = os.path.join(backup_root, f"{doc_id}-{token}")
        os.makedirs(staging)

        rows: list[tuple] = []
        seen_assets: dict[str, str] = {}
        for asset in assets:
            if not ASSET_ID_PATTERN.fullmatch(asset.asset_id):
                raise ValueError(f"Unsafe PDF asset identifier: {asset.asset_id!r}")
            # PDFs may reference the same embedded image more than once on a
            # page.  Its content-addressed ID remains the provenance key, so
            # store it once rather than failing the entire reindex on a second
            # identical image.  Different bytes under one ID would indicate a
            # parser bug and must remain a hard failure.
            prior_digest = seen_assets.get(asset.asset_id)
            if prior_digest is not None:
                if prior_digest != asset.sha256:
                    raise ValueError(f"Conflicting PDF asset payload for {asset.asset_id!r}")
                continue
            seen_assets[asset.asset_id] = asset.sha256
            filename = f"{asset.asset_id}{asset.extension}"
            target = os.path.join(staging, filename)
            with open(target, "xb") as file:
                file.write(asset.data)
            rows.append((
                asset.asset_id,
                doc_id,
                asset.page_number,
                json.dumps(asset.bounding_box) if asset.bounding_box else None,
                filename,
                asset.mime_type,
                asset.sha256,
                asset.caption,
                asset.width,
                asset.height,
                len(asset.data),
            ))
        return cls(root, doc_id, staging, active, backup, rows)

    def promote(self) -> None:
        if os.path.isdir(self.active):
            os.replace(self.active, self.backup)
        os.replace(self.staging, self.active)
        self.promoted = True

    def rollback(self) -> None:
        if self.promoted and os.path.isdir(self.active):
            shutil.rmtree(self.active)
        if os.path.isdir(self.backup):
            os.replace(self.backup, self.active)
        if os.path.isdir(self.staging):
            shutil.rmtree(self.staging)
        self.promoted = False

    def finalize(self) -> None:
        if os.path.isdir(self.backup):
            shutil.rmtree(self.backup)
        if os.path.isdir(self.staging):
            shutil.rmtree(self.staging)


def asset_path(data_dir: str, doc_id: str, filename: str) -> str:
    root = os.path.abspath(os.path.join(data_dir, "document-assets", doc_id))
    resolved = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError("Unsafe document asset path.")
    return resolved


def delete_document_assets(data_dir: str, doc_id: str) -> None:
    root = os.path.abspath(os.path.join(data_dir, "document-assets"))
    target = os.path.abspath(os.path.join(root, doc_id))
    if os.path.commonpath([root, target]) != root or target == root:
        raise ValueError("Unsafe document asset directory.")
    if os.path.isdir(target):
        shutil.rmtree(target)
