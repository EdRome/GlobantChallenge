import os
import xml.etree.ElementTree as ET
import pandas as pd

class XMLProcessor:
    """
    Class dedicated to process XML, the input must be an iterator, so the processing is the most memory efficient as possible.
    The process method will read the input iterator each time is called, when the iterator is empty, it will print a message.
    """
    
    def __init__(self, batches: iter, xml_dir: str):
        self.__batches = batches
        self.__xml_dir = xml_dir
    
    def filter_xml_element(self, file_path):
        """
        Filter the tasg of interest. This function can be modified to get any other tag of interest.
        However, changing this implies to modify the convert_to_dataframe function and the get_text_elements
        in case the new tag needs further processing.
        """
        tags_of_interest = {'AwardTitle','Organization','AbstractNarration'}
        context = ET.iterparse(file_path)
        return filter(lambda xml: xml[1].tag in tags_of_interest, context)

    def process_files(self, xml_file):
        """
        Maps the filtering functions to each xml.
        """
        return map(lambda file: self.filter_xml_element(file), xml_file)

    def prepare_files(self, lst):
        """
        The input list is full of names only. This function concat the 
        XML directory to the filename.
        """
        return map(lambda file: os.path.join(self.__xml_dir, file), lst)

    def get_text_elements(self, xml):
        """
        It get the text data from the filtered tags. Only organization recieves a particular
        case since the function looks for Division - LongName.
        """
        if xml[1].tag == 'Organization':
            # Search for Division tag inside Organization
            division = [elem for elem in xml[1] if elem.tag == 'Division']
            # Inside Division, search to LongName (the name of the division)
            long_name = [elem for elem in division[0] if elem.tag == 'LongName']
            return long_name[0].text
        else:
            return xml[1].text

    def get_data_from_generator(self, data):
        """
        It maps the get_text_element to generate a list of text data
        """
        return [map(lambda x: self.get_text_elements(x), elem) for elem in data]

    def convert_to_dataframe(self, text_data):
        """
        Conver the input text in list format into a dataframe for further processing
        """
        return pd.DataFrame(
            data=[list(data) for data in text_data],
            columns=['AwardTitle','Division','AbstractNarration']
        )

    def process(self):
        """
        Main function of the class
        """
        try:
            lst = next(self.__batches)

            xml_files = self.prepare_files(lst)
            xml_data = self.process_files(xml_files)
            xml_text_data = self.get_data_from_generator(xml_data)
            return self.convert_to_dataframe(xml_text_data).assign(file = lst)
        except StopIteration:
            print("Iterator is empty")