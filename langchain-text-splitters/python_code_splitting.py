from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Dog:
    # Class attribute (shared by all instances)
    species = "Canine" 
    
    # Constructor / Instance attributes (unique to each instance)
    def __init__(self, name, age):
        self.name = name  
        self.age = age    
        
    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old."

# Instantiating the object
my_dog = Dog("Buddy", 3)

print(my_dog.description())  # Output: Buddy is 3 years old.
print(my_dog.species)        # Output: Canine
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])