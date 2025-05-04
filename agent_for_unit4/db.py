import shelve
import shutil
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ShelveDB(Generic[T]):
    dir_path: Path

    def __init__(self, db_name: str, init: bool) -> None:
        self.db_path = self.dir_path / db_name

        if init:
            self.dir_path.mkdir(parents=True, exist_ok=True)
            for file_path in self.dir_path.glob(f"{db_name}*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)

    @classmethod
    def from_table(cls, table: str) -> "ShelveDB":
        return cls(table, False)

    def save(self, key: str, value: Any) -> None:
        with shelve.open(str(self.db_path)) as db:
            db[key] = value

    def fetch(self, key: str) -> T | None:
        with shelve.open(str(self.db_path)) as db:
            return db.get(key, None)

    def delete(self, key: str) -> bool:
        with shelve.open(str(self.db_path)) as db:
            if key in db:
                del db[key]
                return True
            return False

    def list_keys(self) -> list[str]:
        with shelve.open(str(self.db_path)) as db:
            return list(db.keys())
