import librosa
import numpy as np

# Specify the song name here
song = "The American Civil War"

def scale_values(values, new_min=0, new_max=32):
    values = np.log1p(values - np.min(values))
    old_min = np.min(values)
    old_max = np.max(values)
    scaled_values = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_values

def calculate_band_averages(y, sr, bands, n_fft=2048, hop_length=4800):
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    bin_ranges = [(int(np.floor(f1 * n_fft / sr)), int(np.floor(f2 * n_fft / sr))) for f1, f2 in bands]
    levels = [[] for _ in range(len(bands))]
    for t in range(D.shape[1]):
        for i, (bin_start, bin_end) in enumerate(bin_ranges):
            level_data = D[bin_start:bin_end, t]
            levels[i].append(np.mean(level_data) if level_data.size > 0 else 0)
    return np.array(levels).T

def array_to_lua_table(arr, table_name, decimal_places=2, values_per_line=60):
    lua_table = f"{table_name} = {{\n"
    for i, item in enumerate(arr.T):
        lua_table += f"\tLevel{i+1} = {{\n"
        for j, val in enumerate(item):
            lua_table += f"{int(round(val)) if decimal_places == 0 else round(val, decimal_places)}, "
            if (j + 1) % values_per_line == 0:
                lua_table += "\n"
        lua_table = lua_table.rstrip(", ") + "\n\t},\n"
    lua_table += "}"
    return lua_table

def generate_lua_script(file_path, bands):
    y, sr = librosa.load(file_path, sr=None)
    band_averages = calculate_band_averages(y, sr, bands)
    scaled_band_averages = scale_values(band_averages)
    table_name = "tblSong"
    lua_table = array_to_lua_table(scaled_band_averages, table_name, 0)

    lua_script = f"""
{lua_table}

local module = {{}}

local tbl = {{}}

table.insert(tbl, {table_name})

local minSamples = math.huge
for k, v in next, tbl do
    for k1, v1 in next, v do
        if #v1 < minSamples then
            minSamples = #v1
        end
    end
end
table.insert(tbl, minSamples)

function module.returnAudioFile()
    return tbl
end

return module
"""

    output_file_path = f"{song}_data.lua"
    with open(output_file_path, "w") as file:
        file.write(lua_script)

# Example usage
bands = [
    (16, 60),
    (60, 250),
    (250, 500),
    (500, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 20000)
]

# Specify the path to your audio file here
file_path = f"{song}.ogg"

generate_lua_script(file_path, bands)
