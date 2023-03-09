

def binary_search_max(f, min_value, max_value):
    l, r = min_value, (max_value + 1)
    
    while l < r:
        mid = l + (r - l) // 2
        
        if f(mid):
            l = mid + 1
        else:
            r = mid

    return (l - 1)


def binary_search_min(f, min_value, max_value):
    l, r = min_value, (max_value + 1)
    
    while l < r:
        mid = l + (r - l) // 2
        
        if f(mid):
            r = mid
        else:
            l = mid + 1

    return l
