package cc.factorie.app.nlp.phrase
import cc.factorie.app.nlp._

class PosBasedNounPhrase(section:Section, start:Int, length:Int, headTokenOffset: Int = -1) extends NounPhrase(section, start, length)

/** Find and chunk noun phrases merely by contiguous nouns (possibly prefixed by adjectives) and pronouns. */
object NounPhraser1 extends DocumentAnnotator {
  def process(document:Document): Document = {
    val phrases = new NounPhraseList
    var tempSpan: NounPhrase = null
    for (section <- document.sections; token <- section.tokens) {
      // Put a span around contiguous sequences of NN or PR part-of-speech prefixes
      val posPrefix = token.attr[pos.PennPosLabel].categoryValue.take(2)
      if (posPrefix == "NN" || posPrefix == "PR" || (posPrefix == "JJ" && token.hasNext && token.next.attr[pos.PennPosLabel].categoryValue.take(2) == "NN")) {
        if (tempSpan eq null) tempSpan = new PosBasedNounPhrase(section, token.position, 1)
        else tempSpan.append(1)(null)
      } else if (tempSpan ne null) {
        if (token.string == "-" && token.hasNext && token.next.attr[pos.PennPosLabel].categoryValue.take(2) == "NN") tempSpan.append(1)(null) // Handle dashed nouns
        else { phrases += tempSpan; tempSpan = null}
      }
    }
    document.attr += phrases
    document
  }
  override def tokenAnnotationString(token:Token): String = {
    val phrases = token.document.attr[NounPhraseList].spansContaining(token)
    if (phrases.isEmpty) return null
    phrases.map(c => if (c.head == token) "B-NP" else "I-NP").mkString(",")
  }
  def prereqAttrs: Iterable[Class[_]] = List(classOf[pos.PennPosLabel])
  def postAttrs: Iterable[Class[_]] = List(classOf[PosBasedNounPhrase])
}
