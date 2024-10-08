import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('best.pt')

# Define class names
classnames = [
    'antelope', 'bear', 'cheetah', 'human', 'coyote', 'crocodile', 
    'deer', 'elephant', 'flamingo', 'fox', 'giraffe', 'gorilla',
    'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird',
    'hyena', 'kangaroo', 'koala', 'leopard', 'lion', 'meerkat',
    'mole', 'monkey', 'moose', 'okapi', 'orangutan', 
    'ostrich', 'otter', 'panda', 'pelecaniformes', 
    'porcupine', 'raccoon', 'reindeer', 'rhino',
    'rhinoceros', 'snake', 'squirrel', 
    'swan', 'tiger', 'turkey',
    'wolf', 'woodpecker', 
    'zebra'
]

# Dictionary of animal descriptions

animal_descriptions = {
    'antelope': 'Antelopes are swift and graceful creatures, known for their speed and agility.',
    'bear': 'Bears are large mammals with thick fur, and are known for their strength and intelligence.',
    'cheetah': 'Cheetahs are the fastest land animals, capable of reaching speeds up to 60 mph in short bursts.',
    'human': 'Humans are intelligent beings known for their complex social structures and advanced tool use.',
    'coyote': 'Coyotes are adaptable canines found in North America, known for their cleverness and resourcefulness.',
    'crocodile': 'Crocodiles are large reptiles found in tropical regions, recognized for their powerful jaws and stealth.',
    'deer': 'Deer are herbivorous mammals known for their gracefulness and their ability to run swiftly.',
    'elephant': 'Elephants are the largest land animals, known for their intelligence, strong social bonds, and long memories.',
    'flamingo': 'Flamingos are known for their distinctive pink color, long legs, and their unique feeding behavior.',
    'fox': 'Foxes are small to medium-sized canines, recognized for their cunning nature and adaptability.',
    'giraffe': 'Giraffes are the tallest land animals, known for their long necks and unique coat patterns.',
    'gorilla': 'Gorillas are large primates that live in social groups and are known for their intelligence and strength.',
    'hedgehog': 'Hedgehogs are small mammals covered in spines, known for their ability to curl into a ball for protection.',
    'hippopotamus': 'Hippos are large, mostly herbivorous mammals known for their massive size and semi-aquatic lifestyle.',
    'hornbill': 'Hornbills are birds known for their long, curved bills and their unique nesting behavior.',
    'horse': 'Horses are strong, social animals known for their speed, strength, and partnership with humans.',
    'hummingbird': 'Hummingbirds are small birds known for their ability to hover in mid-air and rapid wing beats.',
    'hyena': 'Hyenas are carnivorous mammals known for their distinctive laughter-like vocalizations and scavenging habits.',
    'kangaroo': 'Kangaroos are large marsupials known for their powerful hind legs and unique method of locomotion.',
    'koala': 'Koalas are tree-dwelling marsupials native to Australia, known for their unique diet of eucalyptus leaves.',
    'leopard': 'Leopards are large felines known for their strength, agility, and beautiful spotted coats.',
    'lion': 'Lions are large social cats known as the "king of the jungle," famous for their pride structure.',
    'meerkat': 'Meerkats are small mammals that live in groups and are known for their upright posture and social behavior.',
    'mole': 'Moles are small mammals known for their burrowing habits and velvety fur.',
    'monkey': 'Monkeys are intelligent primates known for their social structures and playful behavior.',
    'moose': 'Moose are large members of the deer family known for their distinctive antlers and long legs.',
    'okapi': 'Okapis are unique mammals related to giraffes, known for their zebra-like stripes on their hindquarters.',
    'orangutan': 'Orangutans are highly intelligent great apes known for their reddish-brown hair and tree-dwelling lifestyle.',
    'ostrich': 'Ostriches are large flightless birds known for their long legs and fast running abilities.',
    'otter': 'Otters are playful aquatic mammals known for their social behavior and use of tools.',
    'panda': 'Pandas are large bears known for their distinctive black and white coloring and diet of bamboo.',
    'pelecaniformes': 'Pelecaniformes are a group of aquatic birds that includes pelicans, herons, and ibises.',
    'porcupine': 'Porcupines are rodents known for their quills, which they use for defense.',
    'raccoon': 'Raccoons are intelligent mammals known for their dexterous front paws and facial mask markings.',
    'reindeer': 'Reindeer are domesticated members of the deer family, known for their ability to pull sleds in snowy regions.',
    'rhino': 'Rhinoceroses are large, thick-skinned herbivores known for their horned snouts.',
    'rhinoceros': 'Rhinoceroses are large, thick-skinned herbivores known for their distinctive horns and endangered status.',
    'snake': 'Snakes are elongated reptiles known for their lack of limbs and unique locomotion methods.',
    'squirrel': 'Squirrels are small to medium-sized rodents known for their bushy tails and acorn-hoarding behavior.',
    'swan': 'Swans are large waterfowl known for their graceful swimming and beautiful plumage.',
    'tiger': 'Tigers are the largest big cats, known for their strength, beautiful stripes, and solitary nature.',
    'turkey': 'Turkeys are large birds native to North America, known for their distinctive wattle and fan-shaped tails.',
    'wolf': 'Wolves are social carnivores known for their pack behavior and vocalizations.',
    'woodpecker': 'Woodpeckers are birds known for their pecking behavior and distinctive drumming sounds.',
    'zebra': 'Zebras are equines known for their distinctive black and white stripes, which help in social interactions.'
}



def detect_objects(frame):
    results = model(frame)
    return results

def main():
    # Set the title in black color
    st.markdown("<h1 style='color: black;'>ANIMAL DETECTION USING YOLO</h1>", unsafe_allow_html=True)

    # Set background image
    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://t4.ftcdn.net/jpg/05/06/97/33/360_F_506973329_NCSJOKETJUZjIW8udBA5LWfZ9vep4kV1.jpg');
            background-size: cover;
            background-position: center;
            height: 100vh;
        }
        </style>
        """, unsafe_allow_html=True)

    # Start webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    stframe = st.empty()
    st.text("")  # Placeholder for animal information

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        
        frame = cv2.resize(frame, (640, 480))
        results = detect_objects(frame)

        detected_animal_info = ""

        # Process bounding boxes and display results
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                class_index = int(box.cls[0])

                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{classnames[class_index]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Get animal description
                    detected_animal_info = animal_descriptions.get(classnames[class_index], "No description available.")

        # Convert frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb)
        
        # Display detected animal information
        st.text(detected_animal_info)

        # Break loop on pressing Esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
