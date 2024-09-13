produk = 1000
response = ""
while response != 'text':
    print(
        "-------------------------"
        " Aplikasi cek barang /n"
        "-------------------------"
        "press 1 for checking how many pruduk you have /n"
        "press 2 for call cinter /n"
    )
    response = input("Enter the 1 or 2: ")
    print("you Entred: " , response)
    if response == "1":
        total = int(input("you have"))
        