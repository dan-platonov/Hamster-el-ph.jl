module MaterialPaths
export material_directory, storage_directory, default_data_directory, check_storage_path
const material_directory = abspath(get(ENV, "EPH_MATERIAL_ROOT", pwd()))
const storage_directory = get(ENV, "EPH_STORAGE_ROOT", "")
function inside(path, root)
    relative = relpath(abspath(path), abspath(root))
    return relative != ".." && !startswith(relative, "../")
end
function check_storage_path(path)
    result = abspath(expanduser(path))
    isempty(storage_directory) || inside(result, storage_directory) ||
        error("Output must be below EPH_STORAGE_ROOT=$storage_directory: $result")
    return result
end
function default_data_directory(calc)
    return isempty(storage_directory) ? joinpath(abspath(calc), "data") :
        joinpath(abspath(storage_directory), relpath(abspath(calc), material_directory))
end
end
