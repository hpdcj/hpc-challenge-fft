from math import log2


def create_2d_table (x: int, y: int):
    return [ [None] * y for t in range(x)]

def interpret (message: str) -> [[str]]:
    array = create_2d_table(12, 6)
    lines = message.splitlines()
    tasks, pernode = -1, -1
    for line in lines:
        if "Tasks" in line:
            split = line.split()
            tasks = int(split[1])
            tasks = int(log2(tasks))

            pernode = int(split[7])
            pernode = int(log2(pernode))
        if "Local" in line:
            split = line.split()
            gflops = split[8]
            present = array[tasks][pernode]
            if present is None or float(present) < float(gflops):
                array[tasks][pernode] = gflops

    return array

def print_results (results: [[str]]):
    for tasks in results:
        for gflops in tasks:
            out = gflops if gflops is not None else "x"
            out.replace(".", ",")
            print (out, '\t', end = '')
        print ()

def read_all_lines() -> str:
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return "\n".join(contents)

if __name__ == "__main__":
    message = read_all_lines()
    result = interpret(message)

    print_results(result)