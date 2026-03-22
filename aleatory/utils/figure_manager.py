import json
from itertools import count
from pathlib import Path


class FigureManager:
    def __init__(
        self,
        output_dir="figures",
        formats=("png", "pdf"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05,
        strip_titles=False,
        close_after_save=False,
        prefix="figure",
        enabled=False,
    ):
        self.output_dir = Path(output_dir)
        self.prefix = prefix

        self.figure_dir = self.output_dir / self.prefix
        self.metadata_dir = self.figure_dir / "metadata"

        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.formats = tuple(formats)
        self.dpi = dpi
        self.bbox_inches = bbox_inches
        self.pad_inches = pad_inches
        self.strip_titles = strip_titles
        self.close_after_save = close_after_save
        self.enabled = enabled
        self._counter = count(1)


    def _next_stem(self):
        return f"{self.prefix}_{next(self._counter):03d}"


    def _strip_titles(self, fig):
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")

        for ax in fig.axes:
            ax.set_title("")


    def _metadata_payload(self, fig, name, formats):
        payload = {
            "name": name,
            "formats": list(formats),
            "figure_manager": {
                "output_dir": str(self.output_dir),
                "figure_dir": str(self.figure_dir),
                "metadata_dir": str(self.metadata_dir),
                "dpi": self.dpi,
                "bbox_inches": self.bbox_inches,
                "pad_inches": self.pad_inches,
                "strip_titles": self.strip_titles,
                "close_after_save": self.close_after_save,
                "prefix": self.prefix,
            },
        }

        figure_metadata = getattr(fig, "_aleatory_metadata", None)
        if figure_metadata is not None:
            payload["figure_metadata"] = figure_metadata

        return payload


    def _write_metadata_json(self, fig, name, formats):
        payload = self._metadata_payload(fig, name=name, formats=formats)
        metadata_path = self.metadata_dir / f"{name}.json"
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metadata_path


    def save(self, fig, name=None, formats=None, strip_titles=None, close=None):
        if not self.enabled:
            return {
                "figure_paths": [],
                "metadata_path": None,
                "saved": False,
            }

        if name is None:
            name = self._next_stem()

        formats = self.formats if formats is None else tuple(formats)
        strip_titles = self.strip_titles if strip_titles is None else strip_titles
        close = self.close_after_save if close is None else close

        if strip_titles:
            self._strip_titles(fig)

        paths = []
        for ext in formats:
            path = self.output_dir / f"{name}.{ext}"
            fig.savefig(
                path,
                dpi=self.dpi,
                bbox_inches=self.bbox_inches,
                pad_inches=self.pad_inches,
            )
            paths.append(path)

        if close:
            import matplotlib.pyplot as plt
            plt.close(fig)

        return {
            "figure_paths": paths,
            "metadata_path": None,
            "saved": True,
        }


    def save_current(self, name=None, formats=None, strip_titles=None, close=None):
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        return self.save(
            fig,
            name=name,
            formats=formats,
            strip_titles=strip_titles,
            close=close,
        )
