"""Registry of the public datasets NeuroChain uses.

Each entry records where a file comes from, how big it is and, where the
publisher or a first download gives one, its SHA-256. Nothing here downloads
anything; see :mod:`neurochain.data.download`.

Access notes (checked 2026-11-13):

* Neuralink removed the compression-challenge files from their bucket, so the
  Internet Archive snapshot is the working source. Its hash is pinned from that
  snapshot. There is no surviving publisher checksum to compare against.
* Dryad publishes SHA-256 digests through its open metadata API but only serves
  file bytes to requests carrying a bearer token.
* Deep Blue Data sits behind an interactive Cloudflare challenge, so its files
  are fetched by hand in a browser and then verified here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFile:
    """One file of a dataset.

    Attributes:
        name: File name under ``data/raw/<dataset key>/``.
        urls: Download URLs tried in order. Empty means browser download only.
        sha256: Expected hex digest, or ``None`` when no digest is known yet.
        size_bytes: Expected size, or ``None`` when unknown.
    """

    name: str
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class Dataset:
    """A dataset: its files plus what a person needs to fetch it by hand.

    Attributes:
        key: Short identifier used on the command line and for the directory.
        title: Human-readable title.
        landing_url: Page describing the dataset.
        licence: Licence as stated by the publisher.
        files: The files that make up the dataset.
        used_in: Build units that consume it.
        token_env: Environment variable holding a bearer token, if the host
            requires one for scripted downloads.
        manual_steps: Instructions shown when a scripted download is impossible.
        zip_member_count: For a single-zip dataset, the number of files expected
            after extraction (used to detect an incomplete unpack).
    """

    key: str
    title: str
    landing_url: str
    licence: str
    files: tuple[RemoteFile, ...]
    used_in: str
    token_env: str | None = None
    manual_steps: str | None = None
    zip_member_count: int | None = None


NEURALINK = Dataset(
    key="neuralink",
    title="Neuralink Compression Challenge: 1 h of raw N1 electrode recordings",
    landing_url="https://content.neuralink.com/compression-challenge/README.html",
    licence="Not stated by the publisher",
    used_in="U0, U2, U5, U7",
    files=(
        RemoteFile(
            name="data.zip",
            urls=(
                "https://content.neuralink.com/compression-challenge/data.zip",
                # `id_` asks the Wayback Machine for the original bytes, unmodified.
                "https://web.archive.org/web/20250804221732id_/"
                "https://content.neuralink.com/compression-challenge/data.zip",
            ),
            sha256="a2288c85158f05eaf29b4d475c1128e061256511e65322090608ec90c2c2432a",
            size_bytes=65_414_611,
        ),
    ),
    zip_member_count=743,
)

NOMAD = Dataset(
    key="nomad",
    title="Data from: Stabilizing brain-computer interfaces through alignment of latent dynamics",
    landing_url="https://datadryad.org/dataset/doi:10.5061/dryad.q83bk3jtp",
    licence="CC0 1.0",
    used_in="U4, U6",
    token_env="DRYAD_TOKEN",
    manual_steps=(
        "Dryad only serves files to a browser or to a request with an API token.\n"
        "  Browser: open the landing page, download session1.nwb and session2.nwb,\n"
        "           and move them into the directory shown above.\n"
        "  Token:   create a free Dryad account, generate an API token, then run\n"
        "           DRYAD_TOKEN=<token> make data"
    ),
    files=(
        RemoteFile(
            name="session1.nwb",
            urls=("https://datadryad.org/api/v2/files/3993226/download",),
            sha256="5c3237688a49d9bdfa034f9715d4ab89cd105c933e41d4c828717b3f9c8a1123",
            size_bytes=104_323_928,
        ),
        RemoteFile(
            name="session2.nwb",
            urls=("https://datadryad.org/api/v2/files/3993227/download",),
            sha256="8d203364522ae67c93cc2de60b1c5349657a082b653d757042463cf55bda2b8a",
            size_bytes=142_855_288,
        ),
    ),
)

NASON = Dataset(
    key="nason",
    title=(
        'Example Data for "A Low-Power Band of Neuronal Spiking Activity Dominated by '
        'Local Single Units Improves the Performance of Brain-Machine Interfaces"'
    ),
    landing_url="https://deepblue.lib.umich.edu/data/concern/data_sets/5712m681k",
    licence="CC BY-NC 4.0",
    used_in="U2, U4",
    manual_steps=(
        "Deep Blue Data requires a browser (Cloudflare challenge).\n"
        "  Open the landing page, download the three .mat files (74 MB in total),\n"
        "  and move them into the directory shown above. The publisher gives no\n"
        "  checksum; the hashes pinned here are those of the copy downloaded on\n"
        "  2026-11-13, so a mismatch means the file differs from that copy."
    ),
    files=(
        RemoteFile(
            name="MultiUnitSim.mat",
            sha256="0f5ea647ee5233dfe6e3368fa6e9ffc13f4eab2bda84abd194b39035c1da3f99",
            size_bytes=18_783_744,
        ),
        RemoteFile(
            name="MultiUnitInVivo.mat",
            sha256="fac792eb8b21d9f443e2004a31d4eb4859b27bca7951883568b9fe33af50a911",
            size_bytes=1_249_657,
        ),
        RemoteFile(
            name="OnlineTrainingData.mat",
            sha256="8b761b9cdf89832e12d324e515858e5095ae09f92e899cf9f381c3e2efd5a50a",
            size_bytes=57_987_417,
        ),
    ),
)

DATASETS: dict[str, Dataset] = {d.key: d for d in (NEURALINK, NOMAD, NASON)}


def get_dataset(key: str) -> Dataset:
    """Look up a dataset by key.

    Raises:
        KeyError: If ``key`` is not registered. The message lists valid keys.
    """
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}") from None
