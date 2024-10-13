import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    total_pages = len(corpus)
    linked_pages = corpus[page]
    
    # Case where the page has no outgoing links, treat it as linking to all pages
    if not linked_pages:
        linked_pages = corpus.keys()
    
    for p in corpus:
        if p in linked_pages:
            distribution[p] = (damping_factor / len(linked_pages)) + ((1 - damping_factor) / total_pages)
        else:
            distribution[p] = (1 - damping_factor) / total_pages
    
    return distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    pages = list(corpus.keys())
    
    # Start with a random page
    current_page = random.choice(pages)
    
    for _ in range(n):
        page_rank[current_page] += 1
        transition = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(pages, weights=transition.values(), k=1)[0]
    
    # Normalize the rank values to sum to 1
    total_samples = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_samples
    
    return page_rank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    page_rank = {page: 1 / total_pages for page in corpus}
    new_page_rank = page_rank.copy()
    
    while True:
        for page in corpus:
            total_sum = 0
            for possible_page in corpus:
                if page in corpus[possible_page]:
                    total_sum += page_rank[possible_page] / len(corpus[possible_page])
                elif len(corpus[possible_page]) == 0:
                    total_sum += page_rank[possible_page] / total_pages
            
            new_page_rank[page] = (1 - damping_factor) / total_pages + damping_factor * total_sum
        
        # Check if ranks have converged
        if all(abs(new_page_rank[page] - page_rank[page]) < 0.001 for page in page_rank):
            break
        
        page_rank = new_page_rank.copy()
    
    # Normalize the rank values to sum to 1
    total_rank = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_rank
    
    return page_rank



if __name__ == "__main__":
    main()
