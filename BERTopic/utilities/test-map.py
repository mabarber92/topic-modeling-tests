from multiprocessing import Pool

def multiplier(l1):
    return l1 * 2


if __name__ == "__main__":
    listed1 = list(range(0, 300000, 2))
    listed2 = list(range(0, 600000, 4))

    print(len(listed1))
    print(len(listed2))



    with Pool(processes=4) as pool:
        multiple_results = [pool.apply_async(multiplier, (item,)) for item in listed1]
        print([res.get(timeout=1) for res in multiple_results])