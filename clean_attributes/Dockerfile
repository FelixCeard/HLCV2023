FROM felixceard/v0:latest

# create team_017 user
WORKDIR /home/hlcv_team017/HLCV2023/
COPY . /home/hlcv_team017/HLCV2023/clean_attributes

#lens_attributes_dictionary.txt
RUN useradd -u 8877 hlcv_team017

RUN mkdir /home/hlcv_team017/HLCV2023/clean_attributes/ExplaiNN
RUN mkdir /home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra

RUN chmod ugo+rwx /home/hlcv_team017
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/clean_attributes
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/clean_attributes/ExplaiNN
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra

WORKDIR /home/hlcv_team017/HLCV2023/clean_attributes

USER hlcv_team017
ENTRYPOINT ["python3.9", "find_knowledge_db_with_explainn.py"]
