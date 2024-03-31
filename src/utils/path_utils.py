from pathlib import Path

from constants import PROJECT_NAME

from typing import Union
from types_.errors import raises_exception


class ProjPaths:
    src_name = "src"
    data_name = "data"
    logs_name = "logs"

    @staticmethod
    def __append_return(path: Path, append_with: Union[Path, bool]) -> Path:
        if append_with:
            return ProjPaths.validate_path(path / append_with)
        return ProjPaths.validate_path(path)

    @raises_exception(FileNotFoundError)
    @staticmethod
    def validate_path(path: Path) -> Union[Path, None]:
        if not path.exists():
            if path.is_dir():
                print(f"Dir path doesn't exist. Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"File {path} does not exist")
        return path

    @staticmethod
    def get_proj_root(append_with: Union[Path, bool] = False) -> Path:
        proj_path_parts = Path(__file__).resolve().parent.parts
        proj_dir_index = next(
            (i for i, part in enumerate(proj_path_parts) if part == PROJECT_NAME), None
        )
        if proj_dir_index is None:
            raise FileNotFoundError(
                f"Could not find project directory with name: {PROJECT_NAME}"
            )

        proj_root_abs_path = Path(*proj_path_parts[: proj_dir_index + 1])
        return ProjPaths.__append_return(proj_root_abs_path, append_with)

    @staticmethod
    def get_src(append_with: Union[Path, bool] = False) -> Path:
        return ProjPaths.__append_return(
            ProjPaths.get_proj_root() / Path(ProjPaths.src_name), append_with
        )

    @staticmethod
    def get_data(append_with: Union[Path, bool] = False) -> Path:
        return ProjPaths.__append_return(
            ProjPaths.get_proj_root() / Path(ProjPaths.data_name), append_with
        )

    @staticmethod
    def get_logs(append_with: Union[Path, bool] = False) -> Path:
        return ProjPaths.__append_return(
            ProjPaths.get_proj_root() / Path(ProjPaths.logs_name), append_with
        )
