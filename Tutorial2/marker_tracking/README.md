Implementação de métodos de "tracking" de marcadores visuais na deteção de articulações do corpo humano

A partir de uma sequência de imagens/video capturadas com uma câmera RGBD (cor + profundidade), é pretendido o desenvolvimento de um algoritmo que implemente soluções de seguimento/localização de marcadores visuais coloridos colocados em vários pontos de interesse no tronco e membros inferiores (cf. vídeo e imagens anexados, 'etc/gait_rgb.avi' e 'etc/gait_depth_frame.png'). 
A deteção em tempo real destes pontos de interesse é de particular relevância na avaliação da postura e da marcha de um sujeito, visto permitir efetuar uma análise do movimento através do cálculo de métricas espacio-temporais, tendo especial valor no seguimento da reabilitação de pacientes com limitações ao nível da marcha (p.e. casos clínicos de ataxia pós-enfarte). Desta forma, um requesito fundamental da solução desenvolvida é a sua capacidade de execução em tempo real, à taxa de aquisição de amostras do sistema de visão (30Hz)

Será fornecida uma solução preliminar funcional que emprega as soluções de tracking já implementadas nas bibliotecas OpenCV (exemplos de deteção nos vídeo 'etc/foot_tracking_example.avi' e 'etc/torso_tracking_example_with_reference.avi'), mas cuja principal limitação é o seu custo computacional, assim como a dificuldade em lidar com oclusões esporádicas dos marcadores visuais.
Existindo na literatura várias abordagens possíveis para resolver o problema (mean/camshift, optical flow, template matching, etc) de tracking visual de marcadores, é inicialmente necessário um breve estudo sobre as mesmas, comparando o seu custo computacional, de modo a selecionar qual a solução que cumpra os requisitos estabelecidos. Adicionalmente, será também endereçada a utilização de marcadores reflectores de infra-vermelhos (fornecidos) e da sua deteção/segmentação/tracking utilizando a câmera de infra-vermelhos de câmeras RGBD.

Será fornecido também acesso a um dataset construído a partir de ensaios já conduzidos num andarilho inteligente equipado com duas câmaras RGBD convencionais (tipo Kinect), em vários contextos (direcção/exposição luminosa), a várias velocidades de locomoção. Pretende-se que sejam identificadas e endereçadas as limitações da instalação atual, nomeadamente cor/tamanho/localização dos marcadores, e caso haja necessidade a realização de novos ensaios com uma instalação mais indicada. Será também interessante analisar de que forma o fornecimento de informação ao algoritmo (p.e. dimensões do sujeito) pode ajudar no processo de seguimento dos marcadores.
Para validação será feita uma comparação com um sistema de captura de movimento comercial (XSens). 

Descrição do código/solução atual: 

Desenvolvido em C++, a solução atual está centrada na classe cv::MultiTracker2 (cf. MultiTracker2.hpp/.cpp) que faz a ponte com a classe cv::Tracker fornecida pelas biliotecas OpenCV, instanciando vários marcadores (6 para imagens do tronco, 4 para imagens dos pés), e que chama o método get() da classe cv::MouseCoordinateSaver que guarda as coordenadas dos pixeis selecionados com o cursor/rato.
O utilizador necessita de selecionar manualmente a posição de cada marcador nas imagens iniciais, tendo posteriormente que interagir novamente com a deteção caso exista alguma falha (no caso de oclusão - comum; ou arrastamento - raro). É também possível reinicializar todos os trackers pressionando a tecla 'r' - útil no caso de o marcador seguir outro objecto que não o marcador.
Apesar de ser possível a selecção de vários algoritmos de tracking, foi verificado que os melhores resultados foram obtidos com o tracker CSRT (Correlation Filter with Channel and Spatial Reliability).

Adicionalmente, é também fornecido código que permite o carregamento e sincronização em runtime do dataset construído sobre o qual incidirá o trabalho. Ao instanciar a classe asbgo::vision::TrialData (cf. TrialData.hpp), tanto os dados de cor (vídeos) como os dados de profundidade (imagens) são carregadas e os "time stamps" de cada amostra idenficados, sendo possível obter um conjunto de dados sincronizados através do método TrialData::next().

Nota: caso seja necessário, pode também ser fornecida a documentação da API (doxygen) para a solução desenvolvida.

Ficheiros anexados:

'include/' -> protótipos das classes referidas
'src/' -> implementação das classes referidas
'compile.sh' -> script para compilação
'etc/' -> ficheiros demonstrativos 
