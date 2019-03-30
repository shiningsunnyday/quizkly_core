import views

user_data = {
    "email": "girishkumar@gmail.com",
    "username"="girishkumar",
    "password"="girishisku",
}

test_data = {
    "content": '''
        In molecular biology, DNA replication is the biological process of
        producing two identical replicas of DNA from one original DNA molecule.
        DNA replication occurs in all living organisms acting as the basis for
        biological inheritance. The cell possesses the distinctive property of
        division, which makes replication of DNA essential. DNA is made up of a
        double helix of two complementary strands. During replication, these
        strands are separated. Each strand of the original DNA molecule then
        serves as a template for the production of its counterpart, a process
        referred to as semiconservative replication. As a result of
        semi-conservative replication, the new helix will be composed of an
        original DNA strand as well as a newly synthesized strand.[1] Cellular
        proofreading and error-checking mechanisms ensure near perfect fidelity
        for DNA replication.[2][3] In a cell, DNA replication begins at specific
        locations, or origins of replication, in the genome.[4] Unwinding of DNA
        at the origin and synthesis of new strands, accommodated by an enzyme
        known as helicase, results in replication forks growing bi-directionally
        from the origin. A number of proteins are associated with the
        replication fork to help in the initiation and continuation of DNA
        synthesis. Most prominently, DNA polymerase synthesizes the new strands
        by adding nucleotides that complement each (template) strand. DNA
        replication occurs during the S-stage of interphase. DNA replication
        (DNA amplification) can also be performed in vitro (artificially,
        outside a cell). DNA polymerases isolated from cells and artificial DNA
        primers can be used to start DNA synthesis at known sequences in a
        template DNA molecule. Polymerase chain reaction (PCR), ligase chain
        reaction (LCR), and transcription-mediated amplification (TMA) are
        examples.
    ''',
    "quiz_name": "DNA Replication"
}

test_user = User.objects.create_user(
    email=user_data["email"],
    username=user_data["username"],
    password=user_data["password"]
)
test_app_user = AppUser(user=test_user)
user_sz = UserSerializer(test_user, data=user_data)
if user_sz.is_valid():
    test_user.save()
    test_app_user.save()
user = authenticate(
    email=user_data["email"],
    username=user_data["username"],
    password=user_data["password"]
)
if user is None:
    raise AuthenticationFailed("Username/password invalid.")
else:
    login(request, user)
test_corpus = Corpus(user=test_app_user, content=test_data["content"])
quiz = Quiz(name=quiz_data)
test_corpus.save()
quiz.save()
process_corpus(test_corpus.id, quiz.id)
