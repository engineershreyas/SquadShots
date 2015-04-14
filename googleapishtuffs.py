


import sys
import getopt
import getpass
import atom
import gdata.contacts.data
import gdata.contacts.client


class Contacts(object):
    def __init__(self, email, password):
        self.gd_client = gdata.contacts.client.ContactsClient(source='GoogleInc-ContactsPythonSample-1')
        self.gd_client.ClientLogin(email, password, self.gd_client.source)

    def PrintAllContacts(self):
        query = gdata.contacts.client.ContactsQuery()
        query.max_results = 500
        feed = self.gd_client.GetContacts(q = query)
        for i, entry in enumerate(feed.entry):
            if hasattr(entry.name, 'full_name'):
                print '\n%s %s' % (i+1, entry.name.full_name.text)
                if entry.content:
                    print '    %s' % (entry.content.text)
                    # Display the primary email address for the contact.
                for email in entry.email:
                    print '    %s' % (email.address)

    def email(self, n):
        query = gdata.contacts.client.ContactsQuery()
        query.max_results = 500
        feed = self.gd_client.GetContacts(q = query)
        for i, entry in enumerate(feed.entry):
            if hasattr(entry.name, 'full_name'):
                namep = entry.name.full_name.text
                for email in entry.email:
                    if namep.lower() == n.lower():
                        return email.address
        return 'No contact email found for this person'

def main():
    hi = Contacts(input("Enter Google Username: "),getpass.getpass("Enter Google Password: "))
    #hi.PrintAllContacts()
    print hi.email('Jacob Butler')

if __name__ == '__main__':
  main()
