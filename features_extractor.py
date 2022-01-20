import math
import whois
import requests
from urllib.parse import urlparse,urlencode
from datetime import datetime

class UrlFeaturizer(object):
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.today = datetime.now()

        try:
            self.whois = None  # whois.whois(self.domain)
        except:
            self.whois = None

        try:
            self.response = None #requests.get(self.url)
        except:
            self.response = None

    ## URL string Features
    def entropy(self):
        prob = [float(self.url.count(c)) / len(self.url) for c in dict.fromkeys(list(self.url))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def num_digits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    def url_length(self):
        parse = urlparse(self.url)
        url = parse.netloc + parse.path
        return len(url)

    def num_parameters(self):
        params = urlparse(self.url).query.split('&')
        return len(params)

    def num_fragments(self):
        fragments = urlparse(self.url).fragment.split('#')
        return len(fragments)

    def domain_extension(self):
        ext = self.url.split('.')[-1].split('/')[0]
        return ext

    ## URL domain features

    def has_https(self):
        return 'https:' in self.url

    def url_is_live(self):
        return self.response == 200

    def days_since_registration(self):
        try:
            if self.whois and 'creation_date' in self.whois:
                if isinstance(self.whois['creation_date'], list):
                    creation_date = self.whois['creation_date'][0]
                else:
                    creation_date = self.whois['creation_date']

                diff = self.today - creation_date
                diff = diff.days
                return diff
            else:
                return 0
        except:
            print(self.url)
            print(self.whois)
            return None

    def days_since_expiration(self):
        try:
            if self.whois and 'expiration_date' in self.whois:
                if isinstance(self.whois['expiration_date'], list):
                    expiration_date = self.whois['expiration_date'][0]
                else:
                    expiration_date = self.whois['expiration_date']

                diff = expiration_date - self.today
                diff = str(diff).split(' days')[0]
                return diff
            else:
                return 0
        except:
            print('daysSinceExpiration')
            print(self.url)
            print(self.whois)
            return None

    def have_at_sign(self):
        return 1 if "@" in self.url else 0

    def url_depth(self):
        s = urlparse(self.url).path.split('/')
        depth = 0
        for j in range(len(s)):
            if len(s[j]) != 0:
                depth = depth + 1
        return depth

    def redirection(self):
        parse = urlparse(self.url)
        url = parse.netloc + parse.path
        return 0 if url.rfind('//') == -1 else 1

    def num_dash(self):
        return self.url.count('-')

    def num_under_score(self):
        return self.url.count('_')

    def num_dots(self):
        return self.url.count('.')

    def get_page_rank(self):
        raise NotImplementedError

    def run(self):
        data = {}

        data['entropy'] = self.entropy()
        data['num_digits'] = self.num_digits()
        data['have_at_sign'] = self.have_at_sign()
        data['url_length'] = self.url_length()
        data['num_parameters'] = self.num_parameters()
        data['has_https'] = self.has_https()
        data['url_is_live'] = self.url_is_live()
        data['url_depth'] = self.url_depth()
        data['num_dots'] = self.num_dots()
        data['redirection'] = self.redirection()
        data['num_dash'] = self.num_dash()
        data['num_under_score'] = self.num_under_score()
        data['ext'] = self.domain_extension()
        #         data['dsr'] = self.daysSinceRegistration()
        #         data['dse'] = self.days_since_expiration()
        return data