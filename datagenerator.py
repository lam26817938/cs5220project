import random
import string

def generate_random_string(length):
    """
    Generates a random string of a given length using uppercase, lowercase letters, and digits.

    Parameters:
        length (int): The length of the string to generate.

    Returns:
        str: A random string of the specified length.
    """
    # Combine uppercase letters, lowercase letters, and digits
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def write_to_file(filename, content):
    """
    Writes content to a file.

    Parameters:
        filename (str): The name of the file to write to.
        content (str): The content to write into the file.
    """
    with open(filename, 'w') as file:
        file.write(content)

# Example usage
length_of_string = 16000  # Specify the length of the random string you want
random_string = generate_random_string(length_of_string)
write_to_file('input.txt', random_string)