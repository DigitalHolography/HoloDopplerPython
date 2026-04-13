"""Manual comparison script kept for local experimentation only."""


def main() -> None:
    from holodoppler.Holodoppler import Holodoppler
    from holodoppler.config import load_builtin_parameters
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    parameters = load_builtin_parameters().to_dict()

    print("parameters :", parameters)

    hd = Holodoppler(backend="cupy", pipeline_version="old")

    holo_path = "D:\\STAGE\\260113_AUZ0752_2.holo"

    hd.load_file(holo_path)
    print("file header :", hd.file_header)

    desktop = os.path.join(os.path.join(os.path.expanduser("~")), "Desktop")

    test_path = os.path.join(desktop, "tempold.h5")

    hd.process_moments_(parameters, h5_path=test_path, mp4_path=None, return_numpy=False)

    ref_path = "D:\\STAGE\\260113_AUZ0752_2_HD_2\\raw\\260113_AUZ0752_2_HD_2_output.h5"

    with h5py.File(test_path, "r") as f_test, h5py.File(ref_path, "r") as f_ref:
        test_data = f_test["moment0"][:, :, :]
        ref_data = np.permute_dims(np.flip(f_ref["moment0"][:, 0, :, :], axis=1), axes=(2, 1, 0))

    plt.figure()
    plt.imshow(np.mean(test_data, axis=2))
    plt.colorbar()
    plt.figure()
    plt.imshow(np.mean(ref_data, axis=2))
    plt.colorbar()

    if test_data.shape != ref_data.shape:
        raise ValueError(f"Shape mismatch: {test_data.shape} vs {ref_data.shape}")

    abs_diff = np.abs(test_data - ref_data)

    plt.figure()
    plt.imshow(abs_diff[:, :, 0])
    plt.colorbar()

    with np.errstate(divide="ignore", invalid="ignore"):
        percent_diff = np.where(
            ref_data != 0,
            abs_diff / np.abs(ref_data) * 100,
            0.0,
        )

    print("=== Absolute Difference ===")
    print(f"Min:  {abs_diff.min()}")
    print(f"Max:  {abs_diff.max()}")
    print(f"Mean: {abs_diff.mean()}")

    print("\n=== Percentage Difference (%) ===")
    print(f"Min:  {percent_diff.min()}")
    print(f"Max:  {percent_diff.max()}")
    print(f"Mean: {percent_diff.mean()}")

    plt.show()


if __name__ == "__main__":
    main()
