import os
import requests


def download_card_images():
    # Create static/cards directory if it doesn't exist
    os.makedirs("static/cards", exist_ok=True)

    # Card details
    suits = ["hearts", "diamonds", "clubs", "spades"]
    ranks = [
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "jack",
        "queen",
        "king",
        "ace",
    ]

    # Base URL for card images
    base_url = "https://deckofcardsapi.com/static/img"

    # Download each card image
    for suit in suits:
        for rank in ranks:
            filename = f"{rank}_of_{suit}.png"
            filepath = os.path.join("static/cards", filename)

            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"Skipping {filename} - already exists")
                continue

            # Construct URL (format: 2H.png for 2 of Hearts)
            rank_code = rank[0].upper() if rank != "10" else "0"
            suit_code = suit[0].upper()
            url = f"{base_url}/{rank_code}{suit_code}.png"

            try:
                response = requests.get(url)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")

            except Exception as e:
                print(f"Error downloading {filename}: {e}")

    # Download card back image
    back_url = "https://deckofcardsapi.com/static/img/back.png"
    back_filepath = os.path.join("static/cards", "back.png")

    if not os.path.exists(back_filepath):
        try:
            response = requests.get(back_url)
            response.raise_for_status()

            with open(back_filepath, "wb") as f:
                f.write(response.content)
            print("Downloaded card back image")
        except Exception as e:
            print(f"Error downloading card back image: {e}")


if __name__ == "__main__":
    download_card_images()
