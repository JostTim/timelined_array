if __name__ == "__main__":
    import numpy as np

    from timelined_array import TimelinedArray

    AX0_SIZE_3D = 20
    AX1_SIZE_3D = 50
    AX2_SIZE_3D = 75

    TIME_AXIS_3D = 1
    TIME_AXIS_SIZE_3D = locals()[f"AX{TIME_AXIS_3D}_SIZE_3D"]
    ARRAY_SHAPE_3D = (AX0_SIZE_3D, AX1_SIZE_3D, AX2_SIZE_3D)

    AXIS_3D_MAPPING = {0: AX0_SIZE_3D, 1: AX1_SIZE_3D, 2: AX2_SIZE_3D}

    normal_arrray = np.random.rand(*ARRAY_SHAPE_3D)

    timelined_array_3D = TimelinedArray(
        normal_arrray,
        timeline=np.arange(TIME_AXIS_SIZE_3D),
        time_dimension=TIME_AXIS_3D,
    )
    print(timelined_array_3D.shape)
    for index, item in enumerate(timelined_array_3D):
        # print("timelined_array_3D", item)
        pass
    # print("index max of timelined_array_3D: ", index)

    for index, item in enumerate(normal_arrray):
        # print("normal_arrray", item)
        pass
    print("index max of normal_arrray: ", index)
    print("over")
