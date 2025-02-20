import hashlib


def get_file_hash(file_path: str) -> str:
    """
    Вычисляет SHA-256 хеш для файла, загруженного через FastAPI UploadFile.

    Args:
        file_path: Путь к файлу.

    Returns:
        str: SHA-256 хеш файла в виде шестнадцатеричной строки
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:  # Открываем файл в бинарном режиме
        while chunk := f.read(8192):  # Читаем файл по частям
            sha256.update(chunk)  # Обновляем хеш с использованием байтов
    return sha256.hexdigest()
