import hashlib

class BloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _hashes(self, item: str):
        hashes = []
        for i in range(self.hash_count):
            hash_val = int(hashlib.md5((item + str(i)).encode()).hexdigest(), 16)
            hashes.append(hash_val % self.size)
        return hashes

    def add(self, item: str):
        for h in self._hashes(item):
            self.bit_array[h] = 1
    
    def check(self, item: str) -> bool:
        return all(self.bit_array[h] == 1 for h in self._hashes(item))

    def __str__(self):
        return "".join(str(bit) for bit in self.bit_array)


def main():
    # Step 1: Get user input for Bloom Filter setup
    size = int(input("Enter Bloom Filter bit array size (e.g., 20): "))
    hash_count = int(input("Enter number of hash functions (e.g., 3): "))
    bf = BloomFilter(size, hash_count)

    # Step 2: Insert elements
    n = int(input("\nHow many elements do you want to insert? "))
    elements = []
    for _ in range(n):
        item = input("Enter element: ").strip()
        bf.add(item)
        elements.append(item)

    print("Current Bloom Filter bit array:\n", bf)

    # Step 3: Query elements
    q = int(input("\nHow many elements do you want to check? "))
    print("\n🔍 Membership Test Results:")
    for _ in range(q):
        query = input("Enter element to check: ").strip()
        result = bf.check(query)
        if result:
            print(f" - {query} : Possibly Present")
        else:
            print(f" - {query} : Definitely Not Present")


if __name__ == "__main__":
    main()
