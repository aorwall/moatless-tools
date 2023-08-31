def slices(series, length):
    if length <= 0:
        raise ValueError("slice length cannot be zero or negative")
    if not series:
        raise ValueError("series cannot be empty")
    if length > len(series):
        raise ValueError("slice length cannot be greater than series length")
    return [series[i:i+length] for i in range(len(series)-length+1)]
