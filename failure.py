

def take_two(e):
    return e[1]

def reordering(source_track_ids, source_data, target_track_ids):
    t = sorted(source_track_ids)
    # 如果沒有排序 # 就排序
    if t != source_track_ids or len(source_track_ids) != len(target_track_ids) or t != target_track_ids:
        # 如果排序後跟target_track_ids不一樣
        #if t != target_track_ids:
        modify_arr = [(k, v) for (k, v) in enumerate(source_track_ids)]
        modify_arr = sorted(modify_arr, key=take_two)
        print(modify_arr)

        target_data = [None] * len(target_track_ids)

        for i in range(len(target_track_ids)):
            target_data[i] = source_data[modify_arr[i][0]]

        return target_data
    
    return source_data

def main():
    # 人數相同的情況
    person = 3

    source_data = [
        ['a'],
        ['b'],
        ['c'],
    ]

    source_track_ids = [1,2,3]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['a'], ['b'], ['c']])

    person = 3

    source_data = [
        ['a'],
        ['b'],
        ['c'],
    ]

    source_track_ids = [3,1,2]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['b'], ['c'], ['a']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
        ['d']
    ]

    source_track_ids = [1,2,3,4]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['a'], ['b'], ['c']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
        ['d']
    ]

    source_track_ids = [4,2,3,1]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['d'], ['b'], ['c']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
        ['d']
    ]

    source_track_ids = [2, 55, 88, 99]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['a'], ['b'], ['c']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
    ]

    source_track_ids = [1,2,99]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['a'], ['b'], ['c']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
        ['d']
    ]

    source_track_ids = [55, 8, 99, 1]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['d'], ['b'], ['a']])

    person = 3
    source_data = [
        ['a'],
        ['b'],
        ['c'],
        ['d']
    ]

    source_track_ids = [1, 3, 2, 5]
    target_track_ids = [i for i in range(1,person+1)]

    target_data = reordering(source_track_ids, source_data, target_track_ids)

    print(target_data == [['a'], ['c'], ['b']])


main()