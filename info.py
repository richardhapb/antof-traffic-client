from utils import data
import time

if __name__ == "__main__":
    # Se ejecuta cada 5 minutos.
    while True:
        print("Actualizando...")
        data.main()
        print("Actualizado.")
        time.sleep(60 * 5)
