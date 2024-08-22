import xml.etree.ElementTree as ET
import re
import spacy
import inflect
import nltk
import Hydra.config as config

class TreeNode:
    def __init__(self, name, frame):
        self.name = name
        self.frames = set([])
        self.children = {}
        self.has_children = False
        self.parent = None
        
        if frame != None:
            self.add_frame(frame)
    
    def add_frame(self, frame):
        self.frames.add(frame)
        return None

    def add_child(self, child):
        child.parent = self
        self.children[child.name] = child
        self.has_children = True

    def get_child(self, name):
        if name in self.children:
            return self.children[name]
        return None
    
    def __str__(self):
        if len(self.children) == 0:
            return self.name
        else:
            return f"{self.name} -> {str(self.children)}"

    def __repr__(self):
        return f"{self.frames}: {self.__str__()}"

class LexicalUnitManager:
    def __init__(self):
        self.lu_map = {}
        bracket_reg = r" *\[.*\] *"
        parenth_reg = r"\(.*\)"

        self.generics = {"one's":["<PRON>", "<PROPN>"], 
                         "someone's":["<PRON>", "<PROPN>"]}
        self.aliases = {"mm":"millimeter", "cm":"centimeter", "instal":"install", 
                        "could":"can", "couldn't":"can't"}

        self.bracket_re = re.compile(bracket_reg)
        self.parenth_re = re.compile(parenth_reg)

        self.nlp = spacy.load("en_core_web_sm")

        self.num2word = inflect.engine()
        self.stemmer = nltk.stem.SnowballStemmer('english')

    def add_lu(self, name, frame):
        name = ".".join(name.split(".")[:-1])

        # We will make several variants of a single lu name, then add them all to map
        variants = []

        # LU names with hyphens can cause problems for 2 main reasons, misspellings, i.e., brother in law instead of brother-in-law
        # and for tokenization, where brother-in-law becomes 5 tokens ['brother', '-', 'in', '-', 'law']
        if '-' in name:
            # add non-hyphenated version
            variants.append(name.replace("-", " "))
            # split at each 'token'
            variants.append(name.replace("-", " - "))

        # Sometimes lu names have brackets indicating a variant of the base word, i.e., "American [N and S Am]" vs "American"
        # Simply remove the bracketed word and let the model learn to differentiate between the different forms, assuming we have samples
        # These are quite common
        if self.bracket_re.search(name):
            new_name = self.bracket_re.sub("", name).strip()
            # print(f"found brackets, adding new name: {new_name}")
            variants.append(new_name)
        
        # Sometimes lu names have parentheses indicating a variant of the word using some other word, i.e., "pull" vs "pull (someone's) leg"
        # This should be something we are able to pick up on while training, but we need to detect the span as a candidate target,
        # Create a regex for this expression, i.e., "pull * leg"
        # Some of these names have multiple variants in the parenthese, i.e., (in/out) of line, make one for each
        # This is very rare (4 instances total)
        parenths = self.parenth_re.findall(name)
        
        if len(parenths) == 1:
            _name = parenths[0].strip(" ()")
            if "/" in _name:
                for variant in _name.split("/"):
                    # print(f"found multiple in parenth, adding variant: {variant}")
                    variants.append(self.parenth_re.sub(variant, name))
            elif _name in self.generics:
                # Sometimes LUs are in a generic form which doesn't usually appear in text, i.e., the LU "hold (one's) tongue.idio"
                # Here we replace (one's) with the pos tags <PROPN> and <PRON> to allow flexibility in the inputs
                for generic in self.generics[_name]:
                    # print(f"found generic in parenth, adding: {self.parenth_re.sub(generic, name)}")
                    variants.append(self.parenth_re.sub(generic, name))

            else:
                # print(f"found other in parenth, adding: {self.parenth_re.sub(_name, name)}")
                variants.append(self.parenth_re.sub(_name, name))

        # if no variants, add it
        if len(variants) == 0:
            if " " in name:
                words = name.split(" ")
                # print(f"found multi-word lu with no variants, adding: {name} -> {frame}")
                self.insert_into_map(words, frame)
            else: 
                # print(f"found simple lu with no variants, adding: {name}")
                self.insert_into_map(name, frame)

            return
                

        # Add variants
        for variant in variants:
            # If name has space in it, make new node (like tree)
            if " " in variant:
                words = variant.split(" ")
                # print(f"found multi-word lu variant, adding: {words}")
                self.insert_into_map(words, frame)
            else:
                # print(f"found single-word variant, adding: {variant}")
                self.insert_into_map(variant, frame)
            
    def insert_into_map(self, name, frame):
        if isinstance(name, str):
            if name not in self.lu_map:
                self.lu_map[name] = TreeNode(name, frame)
            else:
                self.lu_map[name].add_frame(frame)

        elif isinstance(name, list):
            # print(name)
            if len(name) == 1:
                return self.insert_into_map(name[0], frame)

            if name[0] in self.lu_map:
                root = self.lu_map[name[0]]
            else:
                root = TreeNode(name[0], None)
                self.lu_map[name[0]] = root

            prev_child = root
            new_child = None
            for word in name[1:]:
                if prev_child.get_child(word) != None:
                    # print(f"found child of {prev_child.name} -> {prev_child.get_child(word).name}, continuing")
                    prev_child = prev_child.get_child(word)
                    continue

                new_child = TreeNode(word, None)
                if word == name[-1]:
                    new_child.add_frame(frame)
                prev_child.add_child(new_child)
                prev_child = new_child
            
            # sometimes, because we traverse to the end of the child chain, i.e. have -> to, 
            # we dont actually add the child, so instead add the frame
            if new_child == None:
                prev_child.add_frame(frame)
                    
    def load_lus(self):
        root = ET.parse(f"{config.framenet_path}/luIndex.xml").getroot()
        lu_elements = root.findall(".//{http://framenet.icsi.berkeley.edu}lu")

        for node in lu_elements:
            name = node.attrib["name"]
            frame = node.attrib["frameName"]

            self.add_lu(name, frame)
            if name != name.lower():
                self.add_lu(name.lower(), frame)

        # Some LUs are missed, so add them after
        self.add_extra_lus()

    def add_extra_lus(self):
        self.insert_into_map("but", "Concessive")
        self.insert_into_map("however", "Concessive")
        
        self.insert_into_map("around", "Locative_relation")

        self.insert_into_map("grand", "Dimension")

        self.insert_into_map("maintain", "Activity_ongoing")
        self.insert_into_map("proceed", "Activity_ongoing")

        self.insert_into_map("start", "Setting_fire")

        # Support irregular LU "there be"
        # self.add_lu("there (is/are/was/were).MANUAL", "Existence")
        # self.add_lu("there (has/have/had) been.MANUAL", "Existence")
        # self.add_lu("(is/are) there.MANUAL", "Existence")
        # self.add_lu("there (may/might/could) be.MANUAL", "Existence")

        # Support 've instead of have
        self.add_lu("' ve got on.MANUAL", "Wearing")
        self.add_lu("' ve got.MANUAL", "Possession")
        self.add_lu("' ve got.MANUAL", "Have_associated")
        self.add_lu("' ve on.MANUAL", "Wearing")
        self.add_lu("' ve over.MANUAL", "Have_visitor_over")
        self.add_lu("' ve to do (with).MANUAL", "Cognitive_connection")
        self.add_lu("' ve to.MANUAL", "Required_event")
        self.add_lu("' ve to.MANUAL", "Desirable_event")

        # Support "let X know"
        self.add_lu("let (one's) know.MANUAL", "Telling")

    def find_frames_from_lu(self, lu: str):
        """
        Given an LU, find all frames associated with it. Must be exact match.
        """
        lu = lu.replace('-', ' ').replace('(', '').replace(')','').split('.')[0]
        split_lu = [x for x in lu.split(" ") if len(x) > 0 and x[0] != '[' and x[-1] != ']']

        if len(split_lu) == 0:
            return set([])

        cur_node = self.lu_map.get(split_lu.pop(0), None)

        if cur_node == None:
            return set([])

        while len(split_lu) > 0:
            cur_node = cur_node.get_child(split_lu.pop(0))
            if cur_node == None:
                return set([])
        
        return cur_node.frames

    def lookup_lus(self, sent, return_lus=False):
        def _get_lu_from_node(node):
            _lu_name = node.name
            _cur_node = node
            while _cur_node.parent != None:
                _cur_node = _cur_node.parent
                _lu_name = f"{_cur_node.name} {_lu_name}"
            return _lu_name
        
        def _check_potentials(candidates, potential, tok, name, doc):
            new_potential = []
            pos = f"<{tok.pos_}>".upper() if tok != None else None

            # check if node satisfies a current potential, if not remove potential, and add its frames to that lu
            for i, p in potential:
                if not p.has_children and p.parent == None:
                    candidates.append({"start":i, "end":tok.idx-1 if tok != None else -1, "frames":p.frames, "lu":_get_lu_from_node(p)})
                    continue

                if name in p.children:
                    new_potential.insert(0, (i, p.children[name]))
                elif pos in p.children:
                    new_potential.insert(0, (i, p.children[pos]))
                # elif len(p.frames) > 0:
                #     candidates.append({"start":i, "end":tok.idx-1 if tok != None else -1, "frames":p.frames})
                else: # name not in child, no frames, and has parents? -> check parents
                    # some failed potentials have parents which have frames, so lets find and add them
                    # print("starting:", p.name, p.frames)
                    cur_node = p
                    end_pos = tok.i if tok != None else -1
                    while cur_node != None:
                        # print("Checking:", cur_node.name, cur_node.frames)
                        if len(cur_node.frames) > 0:
                            # found a node with frames, add it and break
                            # print("Adding:", cur_node.name, cur_node.frames)
                            candidates.append({"start":i, "end":doc[end_pos].idx-1, "frames":cur_node.frames, "lu":_get_lu_from_node(cur_node)})
                        end_pos -= 1 # reduce the end counter, +1 for space (assuming its a space, if its not this is messy...)
                        cur_node = cur_node.parent # move to parent
                    
            potential.clear()
            for p in new_potential:
                potential.append(p)

        def _add_candidate(candidates, potential, node, tok):
            # check if node has children, if so make new potential
            if node.has_children:
                # print("adding potential:", tok.text, node.frames)
                potential.append((tok.idx, node))
            else:
                # if no children, add node to candidates
                # print("directly adding:", tok.text)
                candidates.append({"start":tok.idx, "end":tok.idx + len(tok.text), "frames":node.frames, "lu":_get_lu_from_node(node)})

        def _remove_dupes(candidates, return_lus=False):
            new_candidates = {}
            
            if not return_lus: 
                for cand in candidates:
                    _key = (cand["start"], cand["end"])
                    _val = cand["frames"]
                    if _key not in new_candidates:
                        new_candidates[_key] = _val # add if doesnt exist
                    else:
                        new_candidates[_key] = new_candidates[_key].union(_val) # merge if does exist
            else:
                for cand in candidates:
                    _key = (cand["start"], cand["end"])
                    _lu = cand["lu"]
                    _frames = cand["frames"]
                    
                    if _key not in new_candidates:
                        new_candidates[_key] = {_lu:_frames} # add if doesnt exist
                    else:
                        if _lu not in new_candidates[_key]:
                            new_candidates[_key][_lu] = _frames # add if doesnt exist
                        else:
                            new_candidates[_key][_lu] = new_candidates[_key][_lu].union(_frames) # merge if does exist
            
            return new_candidates
        
        def _remove_subsumed(candidates):
            best_candidates = {}
            last_start = -1
            last_end = -1

            for (start, end), frames in sorted(candidates.items(), key=lambda x: x[0]):
                if start > last_end:
                    last_start = start
                    last_end = end
                    best_candidates[(start, end)] = frames

                if start == last_start and end > last_end:
                    best_candidates.pop((last_start, last_end))
                    best_candidates[(start, end)] = frames
                    last_start = start
                    last_end = end

            return best_candidates
        
        def _disambiguate_toks(tok, candidates):
            # Sometimes token pairs can be an LU but we only want the first, i.e., 'kind of' in "what kind of car is that?" vs "it's kind of cold today."
            # We can disambiguate these via pos tags
            _ambiguous_toks = {("kind", "NOUN"):(self.lu_map["kind"].frames, 0, "kind"), 
                               ("kind", "ADV"):(self.lu_map["kind"].get_child("of").frames, 3, "kind of")
                               }

            if prev_tok == None:
                return False

            if (tok.lemma_, tok.pos_) in _ambiguous_toks:
                frames, extra_size, lu = _ambiguous_toks[(tok.lemma_, tok.pos_)]
                candidates.append({"start":tok.idx, "end":tok.idx + len(tok.text) + extra_size, "frames":frames, "lu":lu})
                return True

            return False
                
        def _remove_bad_spans(candidates):
            return {(s,e):v for (s,e),v in candidates.items() if e > s}

        # we process 2 different sentences: 
        # 1, the original sentence, keeping capitalization
        # 2, a cleaned version where certain contractions have been expanded, i.e. 've -> have, and all text is lowercased
        
        candidates = []
        potential_candidates = []
        potential_candidates_full_word = []
        potential_candidates_ignore_pos = []

        prev_tok = None
    
        for doc in self.nlp.pipe([sent, sent.replace("-", " ").lower()], batch_size=2):
            for tok in doc:
                lem = tok.lemma_ # get the lemma which takes into account the POS tags from spacy
                lem_ignore_pos = self.stemmer.stem(tok.text) # get the lemma of single word
                pos = f"<{tok.pos_}>".upper()
                
                # If token is a number, convert it to its word form since there are many LUs which come from numbers
                if tok.pos_ == "NUM":
                    lem = self.num2word.number_to_words(tok.text)
                    lem_ignore_pos = self.num2word.number_to_words(tok.text)

                # if we find disambiguated token, dont add it to potential, just check the previous potentials
                if _disambiguate_toks(tok, candidates):
                    _check_potentials(candidates, potential_candidates, None, None, doc)
                    _check_potentials(candidates, potential_candidates_full_word, None, None, doc)
                    _check_potentials(candidates, potential_candidates_ignore_pos, None, None, doc)
                else:
                    # first, check if current token matches any potential lus
                    _check_potentials(candidates, potential_candidates, tok, lem, doc)
                    _check_potentials(candidates, potential_candidates_full_word, tok, tok.text, doc)
                    _check_potentials(candidates, potential_candidates_ignore_pos, tok, lem_ignore_pos, doc)

                # Check if raw token in lu map

                # print(tok.text, lem, lem_ignore_pos)

                if tok.text in self.lu_map:
                    lu_node = self.lu_map[tok.text]
                    _add_candidate(candidates, potential_candidates_full_word, lu_node, tok)
                
                # Check if lemma in lu map
                if lem in self.lu_map:
                    lu_node = self.lu_map[lem]
                    _add_candidate(candidates, potential_candidates, lu_node, tok)

                    # Check if ancestor or child in parse tree make up the LU
                    for ancestor in tok.ancestors:
                        if ancestor.lemma_ in lu_node.children:
                            if tok.idx > ancestor.idx:
                                candidates.append({"start":ancestor.idx, 
                                                    "end":tok.idx + len(tok.text), 
                                                    "frames":lu_node.children[ancestor.lemma_].frames,
                                                    "lu":_get_lu_from_node(lu_node.children[ancestor.lemma_])})
                            elif tok.idx < ancestor.idx:
                                candidates.append({"start":tok.idx, 
                                                   "end":ancestor.idx + len(ancestor.text), 
                                                   "frames":lu_node.children[ancestor.lemma_].frames,
                                                   "lu":_get_lu_from_node(lu_node.children[ancestor.lemma_])})
                                
                    for child in tok.children:
                        if child.lemma_ in lu_node.children:
                            if tok.idx > child.idx:
                                candidates.append({"start":child.idx, 
                                                   "end":tok.idx + len(tok.text), 
                                                   "frames":lu_node.children[child.lemma_].frames,
                                                   "lu":_get_lu_from_node(lu_node.children[child.lemma_])})
                            elif tok.idx < child.idx:
                                candidates.append({"start":tok.idx, 
                                                   "end":child.idx + len(child.text), 
                                                   "frames":lu_node.children[child.lemma_].frames,
                                                   "lu":_get_lu_from_node(lu_node.children[child.lemma_])})

                # Check if lemma in lu map regardless of pos tag
                if lem_ignore_pos in self.lu_map:
                    lu_node = self.lu_map[lem_ignore_pos]
                    _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)
                
                # Check aliases (in case lemmatizer fails)
                if tok.text in self.aliases:
                    lu_node = self.lu_map[self.aliases[tok.text]]
                    _add_candidate(candidates, potential_candidates_full_word, lu_node, tok)
                
                # Since this world has no consistency, we must check if words are spelt using british or american english :)
                if tok.text in config.gb2us:
                    fixed_english = config.gb2us[tok.text]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)
                elif lem in config.gb2us:
                    fixed_english = config.gb2us[lem]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)
                elif lem_ignore_pos in config.gb2us:
                    fixed_english = config.gb2us[lem_ignore_pos]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)

                # Since this world has no consistency, we must check if words are spelt using british or american english :)
                if tok.text in config.us2gb:
                    fixed_english = config.us2gb[tok.text]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)
                elif lem in config.us2gb:
                    fixed_english = config.us2gb[lem]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)
                elif lem_ignore_pos in config.us2gb:
                    fixed_english = config.us2gb[lem_ignore_pos]
                    if fixed_english in self.lu_map:
                        lu_node = self.lu_map[fixed_english]
                        _add_candidate(candidates, potential_candidates_ignore_pos, lu_node, tok)

                prev_tok = tok
        
        # Final check to remove any remaining potential candidates
        _check_potentials(candidates, potential_candidates, None, None, doc)
        
        # Fix -1 ends
        for c in candidates:
            if c["end"] == -1 and " " in sent[c["start"]:]:
                c["end"] = c["start"] + sent[c["start"]:].index(" ")
            elif c["end"] == -1:
                c["end"] = len(sent)
        
        candidates[:] = [x for x in candidates if x["end"] >= 0] 

        return _remove_bad_spans(_remove_dupes(candidates, return_lus=return_lus))

