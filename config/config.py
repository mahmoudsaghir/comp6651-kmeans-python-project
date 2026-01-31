class Config:
    def __init__(self, file_path):
        self.params = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=")
                self.params[key.strip()] = value.strip()

    def get_string(self, key):
        return self.params[key]

    def get_int(self, key):
        return int(self.params[key])

    def get_double(self, key):
        return float(self.params[key])