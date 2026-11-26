ChequeDetector
COMP 4102 project by Gloria Li and Gillian Gonzales

Takes a photo of a cheque, finds it, straightens it, and makes it black and white so it's easier to read. We only tested on fake/voided cheques.

To install: pip3 install -r requirements.txt

To run: python3 main.py photo.jpg (or a folder of photos)

Results go in output/. The steps of the binarization for each photo go in binarize_steps/.

Options: --method niblack (default is sauvola), --window and --k to tune it (defaults 25 and 0.34), --verbose to see why shapes got accepted or rejected.

For detection stats, put photos in test_images/cheques and test_images/no_cheque

detector.py finds and straightens the cheque, binarize_cheque.py makes it black and white, main.py runs both.
